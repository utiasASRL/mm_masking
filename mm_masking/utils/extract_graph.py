from matplotlib import pyplot as plt
import numpy as np
from vtr_pose_graph.graph import Graph
from vtr_pose_graph.vertex import Vertex
import vtr_pose_graph.graph_utils as g_utils
from sensor_msgs_py.point_cloud2 import read_points
from pylgmath import Transformation


def extract_points_from_vertex(v: Vertex, msg="raw_point_cloud"):
    raw_pc_msg = v.get_data(msg)
    pc = read_points(raw_pc_msg.point_cloud)
    p_this = np.vstack((pc['x'], pc['y'], pc['z'], pc['normal_x'], pc['normal_y'], pc['normal_z'])).astype(np.float32)
    T_v_this = Transformation(xi_ab=np.array(raw_pc_msg.t_vertex_this.xi).reshape(6, 1)).matrix()

    return p_this, T_v_this

def extract_points_and_map(graph: Graph, v: Vertex, msg_prefix='', extract_raw_pts = True):
    # Frame explainer:
    # F_ls: loc sensor frame (e.g. radar in radar-lidar loc)
    # F_map: map frame (global frame in which the map is resolved in, NOT THE MAP SENSOR FRAME)
    # F_r: (teach) robot frame, don't need repeat robot frame so just using r

    filtered_msg = msg_prefix + 'filtered_point_cloud'
    p_ls, _, = extract_points_from_vertex(v, msg=filtered_msg)  # "this" frame is the loc sensor frame
    if extract_raw_pts:
        raw_msg = msg_prefix + 'raw_point_cloud'
        p_ls_raw, _, = extract_points_from_vertex(v, msg=raw_msg)  # "this" frame is the loc sensor frame
    else:
        p_ls_raw = p_ls

    print(p_ls[:,0])

    p_map, T_r_map = extract_points_from_vertex(v, msg="submap_loc")  # "this" frame is the map frame

    teach_v = g_utils.get_closest_teach_vertex(v)
    map_ptr = teach_v.get_data("pointmap_ptr")
    teach_v = graph.get_vertex(map_ptr.map_vid)
    # teach_v = graph.get_vertex(map_ptr.map_vid)
    #map_pts, maps_norms = extract_points_from_vertex(teach_v, msg="pointmap", T_zero=False) # This extracts the full, unfiltered lidar submap!

    # Extract timestamps
    loc_stamp = int(v.stamp * 1e-3)
    map_stamp = int(teach_v.stamp * 1e-3)

    return p_ls_raw.T, p_ls.T, p_map.T, T_r_map, loc_stamp, map_stamp

def extract_T_v_this(v: Vertex, msg='submap_loc'):
    raw_pc_msg = v.get_data(msg)
    return Transformation(xi_ab=np.array(raw_pc_msg.t_vertex_this.xi).reshape(6, 1)).matrix()