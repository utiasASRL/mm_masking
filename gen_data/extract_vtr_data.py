import os
import time

import numpy as np
import matplotlib.pyplot as plt
import vtr_pose_graph
from vtr_pose_graph.vertex import Vertex

from sensor_msgs_py.point_cloud2 import read_points
from pylgmath import Transformation
from vtr_utils.plot_utils import extract_points_from_vertex, extract_points_and_map, downsample, range_crop
import argparse
from vtr_pose_graph.graph import Graph
import pickle
import os.path as osp

from vtr_pose_graph.graph_factory import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator
import vtr_pose_graph.graph_utils as g_utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Verify Point Cloud',
                        description = 'Plots point cloud to verify alignment')
    parser.add_argument('-g', '--graph', default="graph")      # option that takes a value
    args = parser.parse_args()


    sensor = 'radar'
    map_seq = 'boreas-2020-11-26-13-58'
    scan_seq = 'boreas-2020-12-04-14-00'
    dataset_dir = '/home/dli/mm_masking/data'
    result_dir = '/home/dli/ext_proj_repos/radar_topometric_localization/results'

    # Assemble paths
    graph_dir = osp.join(result_dir, sensor, map_seq, scan_seq, 'graph')
    # Result paths
    map_pc_dir = osp.join(dataset_dir, 'pointclouds', sensor, map_seq)
    if not osp.exists(map_pc_dir):
        os.makedirs(map_pc_dir)

    factory = Rosbag2GraphFactory(graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.get_vertex((0,1))

    save_list = []
    v_id_list = []
    for vertex, e in TemporalIterator(v_start):
        if e.from_id == vtr_pose_graph.INVALID_ID:
            continue
        print(f"Processing vertex {vertex.id} with stamp {vertex.stamp}")
        map_pts, maps_norms = extract_points_from_vertex(vertex, msg="pointmap")
        map_stamp = int(vertex.stamp * 1e-3)
        map_id = vertex.id
        map_pc = np.concatenate((map_pts.T, maps_norms.T), axis=1)
        map_pc_file_name = osp.join(map_pc_dir, str(map_stamp) + ".bin")
    
        # Save point cloud to binary file
        map_pc.tofile(map_pc_file_name)