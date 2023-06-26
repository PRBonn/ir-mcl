import os
import json

import numpy as np
from tqdm import tqdm

import rosbag
import rospy

from geometry_msgs.msg import Point, TransformStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import tf.transformations

from .tools_utils import extract_translation_and_rotation_from_matrix, odom_to_matrix


class CARMENToROSBagConverter:
    """
    Constructor for the CARMENToROSBagConverter class.

    Args:
        carmen_file (str): The file path of the carmen format dataset.
        output_bag (str): The directory for saving the output rosbag.
        start_angle (float, optional):
            The minimal angle of the field of view (start_angle * pi). Default is -0.5.
        end_angle (float, optional):
            The maximal angle of the field of view (end_angle * pi). Default is 0.5.

    Attributes:
        carmen_file (str): The file path of the carmen format dataset.
        output_bag (str): The directory for saving the output rosbag.
        start_angle (float): The minimal angle of the field of view (start_angle * pi).
        end_angle (float): The maximal angle of the field of view (end_angle * pi).
        num_beams (int): The number of beams in the range readings.

    Methods:
        convert(self):
            Converts the CARMEN format dataset to a ROS bag file.
        _extract_datas(self):
            Extracts pose, range, and timestamp data from the CARMEN format dataset.
        _generate_rosbag(self, poses_raw, ranges_raw, timestamps):
            Generates a ROS bag file from the extracted pose, range, and timestamp data.
    """
    def __init__(self, carmen_file, output_bag, start_angle=-0.5, end_angle=0.5):
        self.carmen_file = carmen_file
        self.output_bag = output_bag
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.num_beams = None

    def convert(self):
        poses_raw, ranges_raw, timestamps = self._extract_datas()
        self._generate_rosbag(poses_raw, ranges_raw, timestamps)

    def _extract_datas(self):
        poses_raw = []
        ranges_raw = []
        timestamps = []
        with open(self.carmen_file) as dataset:
            for line in dataset.readlines():
                line = line.strip()
                tokens = line.split(' ')

                if tokens[0] == 'FLASER':
                    num_beams = int(tokens[1])
                    self.num_beams = num_beams
                    # timestamps
                    timestamps.append(float(tokens[-1]))

                    # range readings
                    ranges = [float(r) for r in tokens[2:num_beams + 2]]
                    ranges_raw.append(ranges)

                    # odometry reading
                    odom_x, odom_y, odom_theta = [float(odom) for odom in tokens[-6:-3]]
                    poses_raw.append([odom_x, odom_y, odom_theta])

        print("Generated {} scans and poses!".format(len(poses_raw)))

        return poses_raw, ranges_raw, timestamps

    def _generate_rosbag(self, poses_raw, ranges_raw, timestamps):
        # Sensor configuration
        start_angle = self.start_angle * np.pi
        end_angle = self.end_angle * np.pi
        angular_res = np.pi / self.num_beams

        # Create ROS bag
        try:
            bag = rosbag.Bag(self.output_bag, "w")
        except (IOError, ValueError):
            print("Couldn't open %", self.output_bag)
            exit(-1)

        laser_msg = LaserScan()
        laser_msg.header.frame_id = 'laser_link'

        laser_msg.angle_min = start_angle
        laser_msg.angle_max = end_angle
        laser_msg.angle_increment = angular_res
        laser_msg.range_min = 0.0
        laser_msg.range_max = 81.91

        tfmsg = TFMessage()
        odom_msg = Odometry()
        tf_odom_robot_msg = TransformStamped()
        tf_robot_laser_msg = TransformStamped()

        print("Converting to ROS bag......")
        for t, odom, ranges in tqdm(zip(timestamps, poses_raw, ranges_raw), total=len(timestamps)):
            x, y, yaw = odom
            position = Point(x, y, 0.0)
            orientation = tf.transformations.quaternion_from_euler(0, 0, yaw)
            timestamp = rospy.Time.from_sec(t)

            # Create TF tree: odom -> base_line -> laser_link
            tf_robot_laser_msg.header.stamp = timestamp
            tf_robot_laser_msg.header.frame_id = 'base_link'
            tf_robot_laser_msg.child_frame_id = 'laser_link'
            q = tf.transformations.quaternion_from_euler(0, 0, 0)
            tf_robot_laser_msg.transform.rotation.x = q[0]
            tf_robot_laser_msg.transform.rotation.y = q[1]
            tf_robot_laser_msg.transform.rotation.z = q[2]
            tf_robot_laser_msg.transform.rotation.w = q[3]

            tfmsg.transforms.append(tf_robot_laser_msg)

            tf_odom_robot_msg.header.stamp = timestamp
            tf_odom_robot_msg.header.frame_id = 'odom'
            tf_odom_robot_msg.child_frame_id = 'base_link'
            tf_odom_robot_msg.transform.translation = position
            tf_odom_robot_msg.transform.rotation.x = orientation[0]
            tf_odom_robot_msg.transform.rotation.y = orientation[1]
            tf_odom_robot_msg.transform.rotation.z = orientation[2]
            tf_odom_robot_msg.transform.rotation.w = orientation[3]

            tfmsg.transforms.append(tf_odom_robot_msg)

            # Convert range reading to sensor_msgs/LaserScan message
            laser_msg.header.stamp = timestamp
            laser_msg.ranges = ranges

            # Convert odometry reading to nav_msgs/Odometry
            odom_msg.header.stamp = timestamp
            odom_msg.pose.pose.position = position
            odom_msg.pose.pose.orientation.x = orientation[0]
            odom_msg.pose.pose.orientation.y = orientation[1]
            odom_msg.pose.pose.orientation.z = orientation[2]
            odom_msg.pose.pose.orientation.w = orientation[3]
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_link'

            # Store messages to bag file
            bag.write('tf', tfmsg, timestamp)
            bag.write('odom', odom_msg, timestamp)
            bag.write('scan', laser_msg, timestamp)

            laser_msg.header.seq += 1
            odom_msg.header.seq += 1
            tf_robot_laser_msg.header.seq += 1
            tf_odom_robot_msg.header.seq += 1

            tfmsg = TFMessage()

        print("Job Done!")
        bag.close()


class DataToJsonConverter:
    """
    Convert data to JSON format. The data is organized as follows:

    Arguments:
        timestamps (numpy.ndarray, (N)):
            Array of timestamps.
        odom_data (numpy.ndarray, (N, 4, 4)):
            Array of odometry data,
            transformation matrix from base_link to world frame, T_w2b.
        gt_pose_data (numpy.ndarray, (N, 4, 4)):
            Array of ground-truth pose data,
            transformation matrix from base_link to world frame, Tgt_w2b.
        scan_data (numpy.ndarray, (N, num_beams)):
            Array of scan data.
        T_b2l (numpy.ndarray, (4, 4)):
            Transformation matrix from lidar frame to base_link.
        lidar_info (dict):
            Dictionary containing LiDAR configuration information.
        skip_frames (int, optional):
            Number of frames to skip when converting the data. Defaults to 1.

    Attributes:
        timestamps (numpy.ndarray): Array of timestamps.
        odom_data (numpy.ndarray): Array of odometry data.
        gt_pose_data (numpy.ndarray): Array of ground-truth pose data.
        scan_data (numpy.ndarray): Array of scan data.
        T_b2l (numpy.ndarray): Transformation matrix from lidar frame to base_link.
        lidar_info (dict): Dictionary containing LiDAR configuration information.
        json_data (dict): JSON data structure for conversion.

    Methods:
        _initialize_json_data(): Convert LiDAR configuration information to JSON format.
        convert_to_json_mapping(output_file):
            Convert the data to JSON format and save it for mapping.
        convert_to_json_localization(output_file):
            Convert the data to JSON format and save it for localization.
    """

    def __init__(self, timestamps, odom_data, gt_pose_data, scan_data,
                 T_b2l, lidar_info, skip_frames=1):
        assert timestamps.shape[0] == odom_data.shape[0] \
               == gt_pose_data.shape[0] == scan_data.shape[0], \
            'the frames number should be same in timestamps, pose_gt, odoms, and scans'

        self.timestamps = timestamps[::skip_frames]
        self.odom_data = odom_data[::skip_frames]
        self.gt_pose_data = gt_pose_data[::skip_frames]
        self.scan_data = scan_data[::skip_frames]

        self.T_b2l = T_b2l
        self.lidar_info = lidar_info

        self.json_data = self._initialize_json_data()

    def _initialize_json_data(self):
        """
        Convert LiDAR configuration information to JSON format.
        """
        num_beams = self.lidar_info['num_beams']
        angle_min = self.lidar_info['angle_min']
        angle_max = self.lidar_info['angle_max']
        angle_res = self.lidar_info['angle_increment']
        field_of_view = angle_max - angle_min
        max_range = 10

        json_data = {
            "num_beams": num_beams,
            "angle_min": angle_min,
            "angle_max": angle_max,
            "angle_res": angle_res,
            "field_of_view": field_of_view,
            "max_range": max_range,
            "scans": []
        }

        return json_data

    def convert_to_json_mapping(self, output_file):
        """
        Convert the data to JSON format and save it to the specified output file. (For Mapping)

        Arguments:
            output_file (str): Path to the output JSON file.
        """
        total_data = len(self.timestamps)
        with tqdm(total=total_data, desc="Converting to JSON (for Mapping)", unit="scan") as pbar:
            for timestamp, odom, scan in zip(self.timestamps, self.gt_pose_data, self.scan_data):
                T_w2l = odom @ self.T_b2l
                translation, rotation = extract_translation_and_rotation_from_matrix(T_w2l)
                odom_x, odom_y, _ = translation
                _, _, odom_theta = rotation
                transform_matrix = odom_to_matrix(odom_x, odom_y, odom_theta)
                scan_data = {
                    "timestamp": timestamp,
                    "odom": [odom_x, odom_y, odom_theta],
                    "transform_matrix": transform_matrix.tolist(),
                    "range_reading": scan.tolist()
                }
                self.json_data["scans"].append(scan_data)
                pbar.update(1)

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        with open(output_file, 'w') as f:
            json.dump(self.json_data, f, indent=4)

        print("-> Done!")

    def convert_to_json_localization(self, output_file):
        """
        Convert the data to JSON format and save it to the specified output file.
        (For Localization)

        Arguments:
            output_file (str): Path to the output JSON file.
        """
        total_data = len(self.timestamps)
        with tqdm(total=total_data,
                  desc="Converting to JSON (for Localization)", unit="scan") as pbar:
            for timestamp, pose_gt, odom_reading, scan in \
                    zip(self.timestamps, self.gt_pose_data, self.odom_data, self.scan_data):
                translation_gt, rotation = \
                    extract_translation_and_rotation_from_matrix(pose_gt)
                translation_odom, rotation_odom = \
                    extract_translation_and_rotation_from_matrix(odom_reading)

                scan_data = {
                    "timestamp": timestamp,
                    "pose_gt": [translation_gt[0], translation_gt[1], rotation[2]],
                    "odom_reading": [translation_odom[0], translation_odom[1], rotation_odom[2]],
                    "range_reading": scan.tolist()
                }
                self.json_data["scans"].append(scan_data)
                pbar.update(1)

        # save to file
        output_dir = os.path.join(os.path.dirname(output_file))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w') as f:
            json.dump(self.json_data, f, indent=4)

        np.savetxt(os.path.join(output_dir, "b2l.txt"), self.T_b2l)

        print("-> Done!")
