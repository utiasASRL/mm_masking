import os
import os.path as osp
import argparse
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import numpy.linalg as npla
import csv

import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

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

def main(dataset_dir):
  # Extract bag file from directory
  for bag_file in os.listdir(dataset_dir):
    if bag_file.endswith(".db3"):
      full_path_bag_file = osp.join(dataset_dir, bag_file)
      print("Loading bag file:", full_path_bag_file)
      bag_parser = BagFileParser(full_path_bag_file)
      #bev_messages = bag_parser.get_bag_messages("/vtr/bev_scan")
      pc_messages = bag_parser.get_bag_messages("/vtr/filtered_point_cloud")
      #print("Loaded number of bev_scan messages: ", len(bev_messages))
      print("Loaded number of filtered_point_cloud messages: ", len(pc_messages))

      # Save pointcloud to csv file
      pc_file_name = osp.join(dataset_dir, "pc_{}.csv".format(bag_file[:-6]))
      # Overwrite file to blank
      open(pc_file_name, 'w').close()

      # Loop through messages and save timestamp and point cloud to file
      for message in pc_messages:
        # Compute timestamp
        timestamp = message[1].header.stamp.sec + message[1].header.stamp.nanosec * 1e-9
        timestamp = int(timestamp * 1e6)

        # Save point cloud as array of tuples (x, y, z, normal_x, normal_y, normal_z)
        pc_xyz = pc2.read_points(message[1], skip_nans=True, field_names=("x", "y", "z", "normal_x", "normal_y", "normal_z"))

        # Write to file
        with open(pc_file_name, "a+") as f:
          writer = csv.writer(f, delimiter=",")
          for point in pc_xyz:
            writer.writerow([timestamp, point[0], point[1], point[2], point[3], point[4], point[5]])
      
      print("Written to file:", pc_file_name)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # Assuming following path structure:
  # <rosbag name>/metadata.yaml
  # <rosbag name>/<rosbag name>_0.db3
  parser.add_argument('--dataset', default=os.getcwd(), type=str, help='path to pointcloud dataset')
  #parser.add_argument('--path', default=os.getcwd(), type=str, help='path to vtr folder (default: os.getcwd())')

  args = parser.parse_args()

  main(args.dataset)
  #main('/home/dli/mm_masking/data/')