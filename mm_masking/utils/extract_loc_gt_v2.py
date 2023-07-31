import os
import os.path as osp
from pathlib import Path
import argparse
import numpy as np
from pylgmath import se3op
from pyboreas.data.splits import loc_test, loc_reference
from pyboreas.utils.utils import get_inverse_tf, get_closest_index, \
	rotation_error, SE3Tose3, rotToRollPitchYaw
from pyboreas.utils.odometry import read_traj_file2, read_traj_file_gt2, plot_loc_stats
import csv


def get_inverse_tf(T):
  """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
  T2 = T.copy()
  T2[:3, :3] = T2[:3, :3].transpose()
  T2[:3, 3:] = -1 * T2[:3, :3] @ T2[:3, 3:]
  return T2


class BagFileParser():

  def __init__(self, bag_file):
    try:
      self.conn = sqlite3.connect(bag_file)
    except Exception as e:
      print('Could not connect: ', e)
      raise Exception('could not connect')

    self.cursor = self.conn.cursor()

    ## create a message (id, topic, type) map
    topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()

    self.topic_type = {name_of: type_of for id_of, name_of, type_of in topics_data}
    self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}
    self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in topics_data}

  # Return messages as list of tuples [(timestamp0, message0), (timestamp1, message1), ...]
  def get_bag_messages(self, topic_name):
    topic_id = self.topic_id[topic_name]
    rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
    return [(timestamp, deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp, data in rows]


def main(dataset_dir, result_dir, output_dir):
  result_dir = osp.normpath(result_dir)
  odo_input = osp.basename(result_dir)
  loc_inputs = [i for i in os.listdir(result_dir) if (i != odo_input and i.startswith("boreas"))]
  loc_inputs.sort()
  loc_inputs = loc_inputs[:1]
  print("Result Directory:", result_dir)
  print("Odometry Run:", odo_input)
  print("Localization Runs:", loc_inputs)
  print("Dataset Directory:", dataset_dir)

  # dataset directory and necessary sequences to load
  dataset_odo = BoreasDataset(osp.normpath(dataset_dir), [[odo_input]])

  # generate ground truth pose dictionary
  ground_truth_poses_odo = dict()
  for sequence in dataset_odo.sequences:
    # build dictionary
    precision = 1e7  # divide by this number to ensure always find the timestamp
    ground_truth_poses_odo.update(
        {int(int(frame.timestamp * 1e9) / precision): frame.pose for frame in sequence.radar_frames})
  print("Loaded number of odometry poses: ", len(ground_truth_poses_odo))

  for loc_input in loc_inputs:
    # dataset directory and necessary sequences to load
    dataset_loc = BoreasDataset(osp.normpath(dataset_dir), [[loc_input]])

    # generate ground truth pose dictionary
    ground_truth_poses_loc = dict()
    for sequence in dataset_loc.sequences:
      # build dictionary
      precision = 1e7  # divide by this number to ensure always find the timestamp
      ground_truth_poses_loc.update(
          {int(int(frame.timestamp * 1e9) / precision): frame.pose for frame in sequence.radar_frames})

    print("Loaded number of localization poses: ", len(ground_truth_poses_loc))

    loc_dir = osp.join(result_dir, loc_input)

    data_dir = osp.join(loc_dir, "graph/data")
    if not osp.exists(data_dir):
      continue
    print("Looking at result data directory:", data_dir)

    # get bag file
    bag_file = '{0}/{1}/{1}_0.db3'.format(osp.abspath(data_dir), "localization_result")
    parser = BagFileParser(bag_file)
    messages = parser.get_bag_messages("localization_result")

    result_gt = []
    for message in messages:
      test_seq_timestamp_full = int(int(message[1].timestamp) / 1000)
      map_seq_timestamp_full = int(int(message[1].vertex_timestamp) / 1000)
      map_seq_vertex_id = message[1].vertex_id

      if not int(message[1].timestamp / precision) in ground_truth_poses_loc.keys():
        print("WARNING: time stamp not found 1: ", int(message[1].timestamp / precision))
        continue
      if not int(message[1].vertex_timestamp / precision) in ground_truth_poses_odo.keys():
        print("WARNING: time stamp not found 2: ", int(message[1].vertex_timestamp / precision))
        continue

      test_seq_timestamp = int(message[1].timestamp / precision)
      map_seq_timestamp = int(message[1].vertex_timestamp / precision)
      T_test_map_in_radar_gt = get_inverse_tf(
          ground_truth_poses_loc[test_seq_timestamp]) @ ground_truth_poses_odo[map_seq_timestamp]

      # Save ground truth
      T_map_test_in_radar_gt = T_test_map_in_radar_gt.flatten().tolist()[:12]
      result_gt.append([test_seq_timestamp_full, map_seq_timestamp_full] + T_map_test_in_radar_gt)

    output_dir_gt = osp.join(output_dir, "localization_gt", odo_input)
    os.makedirs(output_dir_gt, exist_ok=True)
    with open(osp.join(output_dir_gt, loc_input + ".txt"), "+w") as file:
      writer = csv.writer(file, delimiter=' ')
      writer.writerows(result_gt)
      print("Written to file:", osp.join(output_dir_gt, loc_input + ".txt"))

def check_time_match(pred_times, gt_times):
    assert(len(pred_times) == len(gt_times)), f"pred time {len(pred_times)} is not equal to gt time {len(gt_times)}"
    p = np.array(pred_times)
    g = np.array(gt_times)
    assert(np.sum(p - g) == 0)

def check_ref_time_match(ref_times, gt_ref_times):
    indices = np.searchsorted(gt_ref_times, ref_times)
    p = np.array(ref_times)
    g = np.array(gt_ref_times)
    assert(np.sum(g[indices] - p) == 0), f"{g[indices].shape} and {p.shape}"

def get_T_enu_s1(query_time, gt_times, gt_poses):
    closest = get_closest_index(query_time, gt_times)
    assert(query_time == gt_times[closest]), 'query: {}'.format(query_time)
    return gt_poses[closest]

def eval_local(result_dir, dataset_dir, output_dir, gt_ref_seq, ref_sensor='lidar', test_sensor='lidar', dim=3, plot_dir=None):
    loc_res_dir = osp.join(result_dir, gt_ref_seq, 'localization_result')
    pred_files = sorted([f for f in os.listdir(loc_res_dir) if f.endswith('.txt')])
    gt_seqs = []
    for predfile in pred_files:
      if Path(predfile).stem.split('.')[0] not in os.listdir(dataset_dir):
        raise Exception(f"prediction file {predfile} doesn't match ground truth sequence list")
      gt_seqs.append(Path(predfile).stem.split('.')[0])

    pred_files = pred_files[:1]

    print("Result Directory:", result_dir)
    print("Odometry Run:", gt_ref_seq)
    print("Localization Runs:", pred_files)
    print("Dataset Directory:", dataset_dir)

    gt_ref_poses, gt_ref_times = read_traj_file_gt2(osp.join(dataset_dir, gt_ref_seq, 'applanix', ref_sensor + '_poses.csv'), dim=dim)
    result_gt = []
    for predfile, seq in zip(pred_files, gt_seqs):
      print('Processing {}...'.format(seq))
      pred_poses, pred_times, ref_times, _, _ = read_traj_file2(osp.join(loc_res_dir, predfile))
      gt_poses, gt_times = read_traj_file_gt2(osp.join(dataset_dir, seq, 'applanix', test_sensor + '_poses.csv'), dim=dim)

      # check that pred_times is a 1-to-1 match with gt_times
      check_time_match(pred_times, gt_times)
      # check that each ref time matches to one gps_ref_time
      check_ref_time_match(ref_times, gt_ref_times)

      for jj in range(len(gt_poses)):
        gt_T_enu_s2 = gt_poses[jj]
        gt_T_enu_s1 = get_T_enu_s1(ref_times[jj], gt_ref_times, gt_ref_poses)

        gt_T_s1_s2 = get_inverse_tf(gt_T_enu_s1) @ gt_T_enu_s2

        if gt_times[jj] == 1607108463252261:
          ah = 1
          #print(gt_times[jj])
          #print(ref_times[jj])
          #print(get_inverse_tf(gt_T_s1_s2))
          #fasddsfa
        T_s1_s2_gt = gt_T_s1_s2.flatten().tolist()[:12]
        result_gt.append([gt_times[jj], ref_times[jj]] + T_s1_s2_gt)

      output_dir_gt = osp.join(output_dir, "localization_gt", gt_ref_seq)
      os.makedirs(output_dir_gt, exist_ok=True)
      with open(osp.join(output_dir_gt, predfile), "+w") as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(result_gt)
        print("Written to file:", osp.join(output_dir_gt, predfile))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # THIS DOESNT WORK YET BC VTR3 ISNT LOADED IN... NOT SURE IF NEED TO

  # Assuming following path structure:
  # <rosbag name>/metadata.yaml
  # <rosbag name>/<rosbag name>_0.db3
  parser.add_argument('--dataset', default='/raid/dli/boreas', type=str, help='path to boreas dataset (contains boreas-*)')
  parser.add_argument('--results', default='/home/dli/ext_proj_repos/radar_topometric_localization/results/radar', type=str, help='path to vtr folder')
  parser.add_argument('--output_path', default='/home/dli/mm_masking/data', type=str, help='path to output')
  parser.add_argument('--sensor', default='radar', type=str, help='sensor')

  args = parser.parse_args()

  if args.sensor == 'radar':
    args.dim = 2
  else:
    args.dim = 3

  args.ref_seq = 'boreas-2020-11-26-13-58'

  eval_local(args.results, args.dataset, args.output_path, args.ref_seq, args.sensor, args.sensor, args.dim)