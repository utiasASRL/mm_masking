from matplotlib import pyplot as plt
import numpy as np
from vtr_pose_graph.graph import Graph
from vtr_pose_graph.vertex import Vertex
import vtr_pose_graph.graph_utils as g_utils
from sensor_msgs_py.point_cloud2 import read_points
from pylgmath import Transformation


def extract_points_from_vertex(v: Vertex, msg="raw_point_cloud", T_zero=False):
    raw_pc_msg = v.get_data(msg)
    new_pc = read_points(raw_pc_msg.point_cloud)
    T_m_v = Transformation(xi_ab=np.array(raw_pc_msg.t_vertex_this.xi).reshape(6, 1))
    
    #T_v_m = Transformation(xi_ab=np.zeros((6, 1)))
    # Define normal transform which only has rotation component of T_v_m
    xi_ab_norm = np.zeros((6, 1))
    xi_ab_norm[3:] = np.array(raw_pc_msg.t_vertex_this.xi)[3:].reshape(3, 1)
    T_m_v_norm = Transformation(xi_ab=xi_ab_norm)

    if T_zero:
        T_m_v = Transformation(xi_ab=np.zeros((6, 1)))
        T_m_v_norm = Transformation(xi_ab=np.zeros((6, 1)))

    points = convert_points_to_frame(np.vstack((new_pc['x'], new_pc['y'], new_pc['z'])), T_m_v).astype(np.float32)
    normals = convert_points_to_frame(np.vstack((new_pc['normal_x'], new_pc['normal_y'], new_pc['normal_z'])), T_m_v_norm).astype(np.float32)
    return points, normals

def convert_points_to_frame(pts: np.ndarray, frame: Transformation):
    if pts.shape[0] != 3:
        raise RuntimeError(f"Expecting 3D points shape was {pts.shape}")
    new_points = np.vstack((pts, np.ones(pts[0].shape, dtype=np.float32)))
    new_points = (frame.matrix() @ new_points)
    return new_points[:3, :]

def extract_points_and_map(graph: Graph, v: Vertex, msg_prefix=''):
    raw_msg = msg_prefix + 'raw_point_cloud'
    curr_raw_pts, _, = extract_points_from_vertex(v, msg=raw_msg, T_zero=True)
    filtered_msg = msg_prefix + 'filtered_point_cloud'
    curr_filtered_pts, _, = extract_points_from_vertex(v, msg=filtered_msg, T_zero=True)

    teach_v = g_utils.get_closest_teach_vertex(v)
    map_ptr = teach_v.get_data("pointmap_ptr")
    teach_v = graph.get_vertex(map_ptr.map_vid)
    map_pts, maps_norms = extract_points_from_vertex(teach_v, msg="pointmap", T_zero=False)

    # Extract timestamps
    loc_stamp = int(v.stamp * 1e-3)
    map_stamp = int(teach_v.stamp * 1e-3)

    return curr_raw_pts.T, curr_filtered_pts.T, map_pts.T, maps_norms.T, loc_stamp, map_stamp