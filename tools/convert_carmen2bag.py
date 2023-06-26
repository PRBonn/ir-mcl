import argparse
import numpy as np

import rosbag
import rospy

from geometry_msgs.msg import  Point, TransformStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import  Odometry
from tf2_msgs.msg import TFMessage
import tf.transformations
from tqdm import tqdm

from libs.format_convert_lib import CARMENToROSBagConverter


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--carmen_file', type=str,
                        default='~/ir-mcl/data/intel/intel.gfs.log',
                        help='the file path of carmen format dataset')
    parser.add_argument('--output_bag', type=str,
                        default='~/ir-mcl/data/intel/intel.bag',
                        help='the directory for saving the output rosbag')

    # sensor configuration
    parser.add_argument('--start_angle', type=float, default=-0.5,
                        help='the minimal angle of the field of view (start_angle * pi).')
    parser.add_argument('--end_angle', type=float, default=0.5,
                        help='the maximal angle of the field of view (end_angle * pi).')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    converter = CARMENToROSBagConverter(args.carmen_file, args.output_bag, args.start_angle, args.end_angle)
    converter.convert()