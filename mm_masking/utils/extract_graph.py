import os
import time

import numpy as np
import matplotlib.pyplot as plt
import simple_graph
from simple_graph.vertex import Vertex

from sklearn.neighbors import NearestNeighbors
from sensor_msgs_py.point_cloud2 import read_points
import open3d as o3d
from pylgmath import Transformation
from src.vtr_utils.plot_utils import extract_points_and_map, downsample, range_crop
import argparse


from src.simple_graph.graph_factory import Rosbag2GraphFactory
from src.simple_graph.graph_iterators import TemporalIterator
import src.simple_graph.graph_utils as g_utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Verify Point Cloud',
                        description = 'Plots point cloud to verify alignment')
    parser.add_argument('-g', '--graph', default="graph")      # option that takes a value
    args = parser.parse_args()

    #offline_graph_dir = os.path.join(os.getenv("VTRTEMP"), "lidar/2023-02-13/2023-02-13/long_repeat")
    offline_graph_dir = '/home/dli/ext_proj_repos/radar_topometric_localization/Johann_results/lidar/boreas-2020-11-26-13-58/boreas-2021-01-26-10-59/graph'
    #offline_graph_dir = '/home/dli/ext_proj_repos/radar_topometric_localization/Johann_results/lidar/boreas-2020-11-26-13-58/boreas-2020-11-26-13-58/graph'
    #offline_graph_dir = os.path.join(os.getenv("VTRROOT"), "vtr_testing_lidar", "tmp", args.graph)

    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.get_vertex((1,0))

    sdfadfas

    x = []
    y = []
    live_2_map = []
    map_2_live = []


    first = True
    paused = False
    def toggle(vis):
        global paused
        paused = not paused
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(' '), toggle)
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    vis.poll_events()
    vis.update_renderer()

    radius_of_interest = 20

    for vertex, e in TemporalIterator(v_start):
        if e.from_id == simple_graph.INVALID_ID:
            continue

        last_v = test_graph.get_vertex(e.from_id)

        live_points, new_map_points, new_labels = extract_points_and_map(test_graph, vertex, labels=True)

        print(live_points.shape, new_map_points.shape)

        robot_position = vertex.T_v_w.r_ba_ina().reshape((3,) )

        new_points = range_crop(live_points[~new_labels], robot_position, radius_of_interest)
        new_obstacles = range_crop(live_points[new_labels], robot_position, radius_of_interest)
        live_points = range_crop(live_points, robot_position, radius_of_interest)
        new_map_points = range_crop(new_map_points, robot_position, radius_of_interest)

        old_points, old_map_points = extract_points_and_map(test_graph, last_v)
        robot_position = last_v.T_v_w.r_ba_ina().reshape((3,) )

        old_points = range_crop(old_points, robot_position, radius_of_interest)
        old_map_points = range_crop(old_map_points, robot_position, radius_of_interest)


        new_points = downsample(new_points, grid_size=0.05)
        live_points = downsample(live_points, grid_size=0.05)
        new_obstacles = downsample(new_obstacles, grid_size=0.05)
        old_points = downsample(old_points, grid_size=0.05)
        new_map_points = downsample(new_map_points, grid_size=0.05)
        old_map_points = downsample(old_map_points, grid_size=0.05)

        #print(new_points.shape, new_obstacles.shape)

        x.append(vertex.T_v_w.r_ba_ina()[0])
        y.append(vertex.T_v_w.r_ba_ina()[1])

        pcd.points = o3d.utility.Vector3dVector(np.vstack((new_points, new_obstacles, new_map_points)))
        pcd.paint_uniform_color((0.45, 0.45, 0.45))
        colors = np.asarray(pcd.colors)

        colors[:new_points.shape[0]] = (16/255,166/255,98/255)
        colors[new_points.shape[0]:new_points.shape[0]+new_obstacles.shape[0]] = (0.95, 0.15, 0.45)


        if first:
            first = False
            vis.add_geometry(pcd)
        else:
            vis.update_geometry(pcd)
        t = time.time()
        while time.time() - t < 0.1 or paused:
            vis.poll_events()
            vis.update_renderer()