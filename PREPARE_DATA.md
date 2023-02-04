## IPBLab dataset
Downloading IPBLab dataset from our server:
```shell
cd ir-mcl && mkdir data && cd data
wget https://www.ipb.uni-bonn.de/html/projects/kuang2023ral/ipblab.zip
unzip ipblab.zip
```

For each sequence, we provide :
- seq_{id}.bag: the ROS bag format, include raw odometer reading and raw lidar reading.
- seq_{id}.json: include raw odometer reading, ground-truth poses, and raw lidar reading.
- seq_{id}_gt_pose: the ground-truth poses in TUM format (for evaluation with evo). 

Besides, there are also some configuration files are provided:
- lidar_info.json: the parameters of the 2D LiDAR sensor.
- occmap.npy: the pre-built occupancy grid map.
- b2l.txt: the transformation from the lidar link to robot's base link 

The final data structure should look like
```
data/
├── ipblab/
│   ├── loc_test/
│   │   ├── test1/
│   │   │   ├──seq_1.bag
│   │   │   ├──seq_1.json
│   │   │   ├──seq_1_gt_pose.txt
│   │   ├──b2l.txt
│   ├──lidar_info.json
│   ├──occmap.npy
```

There is one sequence available for the localization experiments now, the full dataset will be released after our dataset paper is published!

## Intel Lab datatse, Freiburg Building 079 dataset, and MIT CSAIL dataset
Downloading these three classical indoor 2D SLAM datasets from our server:
```shell
cd ir-mcl && mkdir data && cd data
wget https://www.ipb.uni-bonn.de/html/projects/kuang2023ral/2dslam.zip
unzip 2dslam.zip
```

For each sequence, we provide :
- train.json: the training set which is used for mapping or train the NOF model.
- val.json: the validation set for evaluating the model during training. 
- test.json: the test set for evaluating the final model.
- occmap.npy: the pre-built occupancy grid map by using training set.

The final data structure should look like
```
data/
├── fr079/
│   ├──occmap.npy
│   ├──train.json
│   ├──val.json
│   ├──test.json
├── intel/
│   ├──...
├── mit/
│   ├──...
```

Here, we provide the converted format of these dataset for ease of use. The raw data could be found on our website: [2D Laser Dataset](https://www.ipb.uni-bonn.de/datasets/). 