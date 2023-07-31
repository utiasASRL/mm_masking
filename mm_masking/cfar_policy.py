import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import ModuleList
import cv2
from torchvision import datasets, transforms
from dICP.ICP import ICP
from radar_utils import load_pc_from_file, cfar_mask, extract_pc, radar_polar_to_cartesian_diff, radar_cartesian_to_polar, radar_polar_to_cartesian
from neptune.types import File

import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            #nn.init.xavier_uniform_(m.bias.data)
            nn.init.zeros_(m.bias)
            #m.weight.data.fill_(0.0)
            #m.bias.data.fill_(0.0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class LearnCFARPolicy(nn.Module):
    def __init__(self, network_input='cartesian', network_output='cartesian',
                 leaky=False, dropout=0.0,
                 float_type=torch.float64, device='cpu', init_weights=True):
        super().__init__()
        self.float_type = float_type
        self.device = device
        self.network_input = network_input
        self.network_output = network_output
        self.leaky = leaky
        self.dropout = dropout

        # Define constant params (need to move to config file)
        self.res = 0.0596   # This is the old resolution!

        channels = [1, 8, 16, 32, 64, 128, 256]
        #channels = [1, 64, 128, 256, 512, 1024]

        self.channels = channels

        self.encoder = ModuleList(
			[self.conv_block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])

        self.decoder = ModuleList(
            [self.conv_block(channels[i+1], channels[i])
                for i in reversed(range(1,len(channels) - 1))])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final_layer = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=1),
            nn.Sigmoid()
        )

        if init_weights:
            self.apply(weights_init)

        
    def conv_block(self, in_channels, out_channels):
        if self.leaky:
            relu_layer = nn.LeakyReLU(0.1)
        else:
            relu_layer = nn.ReLU()
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        modules.append(relu_layer)
        modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        modules.append(relu_layer)
        if self.dropout > 0.0:
            modules.append(nn.Dropout(p=self.dropout))
        return nn.Sequential(*modules)

    def forward(self, fft_data, binary=False, neptune_run=None, epoch=None):
        # Load in CFAR mask
        input_data = fft_data.unsqueeze(1)

        # Encoder
        enc_layers = []
        for i, layer in enumerate(self.encoder):
            enc_layers.append(input_data)
            if i > 0:
                input_data = self.maxpool(input_data)
            input_data = layer(input_data)
        enc_layers.reverse()

        # Decoder
        for i, decoder_layer in enumerate(self.decoder):
            # Load in skip connection
            skip_con = enc_layers[i]
            # Upsample input data to match skip connection
            bi_w = skip_con.shape[2]
            bi_h = skip_con.shape[3]
            bi_upsample = nn.UpsamplingBilinear2d(size=(bi_w, bi_h))
            input_data = bi_upsample(input_data)
            # Now convolve using decoder to reduce channels
            input_data = decoder_layer(input_data)
            # Concatenate with skip connection
            input_data = torch.cat([enc_layers[i], input_data], dim=1)
            # Pass through decoder again
            input_data = decoder_layer(input_data)
        
        logits = self.final_layer(input_data).squeeze(1)
        mask = logits

        if binary:
            mask = torch.where(mask > 0.5, 1.0, 0.0)

        print("Max and min of mask: " + 
              str(np.round(torch.max(mask.detach()).item(), 3)) + ", " + 
              str(np.round(torch.min(mask.detach()).item(), 3)))

        if neptune_run is not None:
            fig = plt.figure()
            plt.imshow(mask[0].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(location='top', shrink=0.5)
            neptune_run["train/mask"].append(fig, step=epoch, name=("mask " + str(epoch)))
            plt.close()

        return mask