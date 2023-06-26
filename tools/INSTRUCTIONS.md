# Related Tools
We provide several tools related to the project.

1. Conversion between different data formats, currently including:
    - CARMEN to ROS bag
    - ROS bag to JSON
    - JSON to ROS bag
2. Occupancy grid mapping using our dataset. (Coming soon!)

## Data Format Conversion

### Install dependencies
```shell
conda activate irmcl
pip install rospkg pycryptodomex gnupg
```

### CARMEN to ROS bag
CARMEN is the data format for the Intel Lab dataset, Freiburg Building 079 dataset, and MIT CSAIL dataset.

#### Prepare the CARMEN files

- Download the dataset from https://www.ipb.uni-bonn.de/datasets/
- For example:
    + Intel Lab dataset: http://www2.informatik.uni-freiburg.de/~stachnis/datasets/datasets/intel-lab/intel.gfs.log.gz
    + Freiburg Building 079 dataset: http://www2.informatik.uni-freiburg.de/~stachnis/datasets/datasets/fr079/fr079-complete.gfs.log.gz
    + MIT CSAIL dataset: http://www2.informatik.uni-freiburg.de/~stachnis/datasets/datasets/csail/csail.corrected.log.gz

#### Convert to ROS bag
To convert a CARMEN file to a ROS bag, use `convert_carmen2bag.py` and follow these commands (replace with your CARMEN file path):

```shell
cd ~/ir-mcl/
python ./tools/convert_carmen2bag.py \
  --carmen_file ~/ir-mcl/data/intel/intel.gfs.log \
  --output_bag ~/ir-mcl/data/intel/intel.bag
```
The output ROS bag includes:

- TF tree: odom -> base_link -> laser_link
- Topic: `odom` ([nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)), which provides the ground-truth robot pose.
- Topic: `scan` ([sensor_msgs/LaserScan](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html)), which contains the range readings from the 2D LiDAR.

### ROS bag  to JSON
To convert a ROS bag to JSON format, the ROS bag must include at least:

1. The ROS topic in [sensor_msgs/LaserScan](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html) type, e.g. `scan` topic;
2. The robot poses, which can be provided through:
    - The ROS topic in [nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html) type, e.g. `odom` topic; Or,
    - The `tf` topic include, odom -> base_link

The conversion depends on the information included in your ROS bag.
The core idea is to extract the 2D LiDAR range readings and robot poses from the bag file,
and then align the poses and range readings based on timestamps.

Here's an example using a previously converted ROS bag:

- To convert a ROS bag to JSON for mapping:
```shell
cd ~/ir-mcl/
python ./tools/convert_bag2json_mapping.py \
  --bag_file ~/ir-mcl/data/intel/intel.bag \
  --output_json ~/ir-mcl/data/intel/intel.json
```
For mapping, the dataset only needs the ground-truth poses of the 2D LiDAR.
Thus, the `odom` label in the JSON file represents the 2D LiDAR's pose.

- To convert a ROS bag to JSON for localization:
```shell
cd ~/ir-mcl/
python ./tools/convert_bag2json_localization.py \
  --bag_file ~/ir-mcl/data/intel/intel.bag \
  --output_json ~/ir-mcl/data/intel/loc_test/seq.json
```
For localization, the sequence requires the ground-truth poses, odometry readings of the robot,
and the transformation from the 2D LiDAR to the robot. Therefore, `pose_gt` represents the ground-truth poses,
`odom_reading` represents the Odometry readings, and a `b2l.txt` file stores the transformation from the 2D LiDAR to the robot.

#### Suggestions
I do not suggest extracting the dataset directly from the ROS bag! The process involved is complex, making it difficult to debug any issues with the data.

In practice, we split the whole process to several steps:

- Step 1: Extract raw data from the ROS bag, including: `odom`, `scan`, and `tf` etc.;
- Step 2: Generate ground-truth poses.
- Step 3: Synchronize the data, including:**gound-truth poses**, **Odometry readings**, **2D LiDAR scans**;
- Step 4: Convert the synchronized data to JSON format.

By processing the data step by step, the entire process becomes easier to debug and helps avoid data errors.