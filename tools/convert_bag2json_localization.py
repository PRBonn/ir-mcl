import argparse

from libs.rosbag_lib import SyncROSBag
from libs.format_convert_lib import DataToJsonConverter


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--bag_file', type=str,
                        default='~/ir-mcl/data/intel/intel.bag',
                        help='the file path of rosbag')
    parser.add_argument('--output_json', type=str,
                        default='~/ir-mcl/data/intel/loc_test/seq.json',
                        help='the directory for saving the output Json file')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    ############## Step 1: Load bag file ##############
    print("-> Loading bag file...")

    # Create an instance of SyncROSBag
    sync_bag = SyncROSBag(args.bag_file)
    timestamps, scan_data, odom_data, T_b2l, lidar_info = sync_bag.get_data()

    print("-> Done!")

    ############## Step 2: Save to JSON ##############
    # We assume the ground truth is the same as the odometry in the example!
    # In practice, you may want to use real ground truth data.
    # ToDO: loading ground truth data from a file
    # ToDO: synchronization ros records with ground truth data

    json_converter = DataToJsonConverter(
        timestamps=timestamps,
        odom_data=odom_data,
        gt_pose_data=odom_data,
        scan_data=scan_data,
        T_b2l=T_b2l,
        lidar_info=lidar_info
    )

    # convert to json for localization
    json_converter.convert_to_json_localization(args.output_json)
