<p align="center">
  <h1 align="center"> IR-MCL: Implicit Representation-Based Online Global Localization </h1>
  <p align="center">
    <a href="https://www.ipb.uni-bonn.de/people/haofei-kuang/"><strong>Haofei Kuang</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/xieyuanli-chen/"><strong>Xieyuanli Chen</strong></a>
    ·
    <a href="https://phd.uniroma1.it/web/TIZIANO-GUADAGNINO_nP1536210_IT.aspx"><strong>Tiziano Guadagnino</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/nicky-zimmerman/"><strong>Nicky Zimmerman</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/jens-behley/"><strong>Jens Behley</strong></a>
    ·
    <a href="https://www.ipb.uni-bonn.de/people/cyrill-stachniss/"><strong>Cyrill Stachniss</strong></a>
  </p>
  <p align="center"><a href="https://www.ipb.uni-bonn.de"><strong>University of Bonn</strong></a>
  </h3>
  <div align="center"></div>
</p>


<p align="center">
<img src="https://user-images.githubusercontent.com/18661888/194111809-4f966ab5-64be-45fc-963b-a6fe0a8c14ed.png" width="700"/>
</p>

### Online localization demo
<p align="center">
<img src="https://user-images.githubusercontent.com/18661888/194112420-f83c2d02-e33b-4e8f-87df-bcaab12641a2.gif" width="800">
</p>


## Abstract
Determining the state of a mobile robot is an essential building block of robot navigation systems. In this paper, we address the problem of estimating the robot’s pose in an indoor environment using 2D LiDAR data and investigate how modern environment models can improve gold standard Monte-Carlo localization (MCL) systems. We propose a neural occupancy field (NOF) to implicitly represent the scene using a neural network. With the pretrained network, we can synthesize 2D LiDAR scans for an arbitrary robot pose through volume rendering. Based on the implicit representation, we can obtain the similarity between a synthesized and actual scan as an observation model and integrate it into an MCL system to perform accurate localization. We evaluate our approach on five sequences of a self-recorded dataset and three publicly available datasets. We show that we can accurately and efficiently localize a robot using our approach surpassing the localization performance of state-of-the-art methods. The experiments suggest that the presented implicit representation is able to predict more accurate 2D LiDAR scans leading to an improved observation model for our particle filter-based localization.


## Dependencies
The code was tested with Ubuntu 20.04 with:
- python version **3.9**.
- pytorch version **1.13.1** with **CUDA 11.6**
- pytorch-lighting with **1.9.0** 

### Installation
- Clone the repo:
  ```shell
  git clone https://github.com/PRBonn/ir-mcl.git
  cd ir-mcl
  ```

- Prepare the python environment (Anaconda is recommended here):
  ```shell
  conda env create -f environment.yml
  ```
  or
  ```shell
  conda create --name irmcl python=3.9.13
  conda activate irmcl
  
  conda install -c conda-forge pybind11
  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
  pip install pytorch-lightning tensorboardX
  pip install matplotlib scipy open3d
  pip install evo --upgrade --no-binary evo
  ```

- Compile the motion model and resampling module
  - Compile the source code:
  ```shell
  cd ir-mcl/mcl & conda activate irmcl
  make -j4
  ```
  - Run the simple Test program:
  ```shell
  cd ir-mcl & conda activate irmcl
  python ./mcl/srrg_utils/test_srrg.py
  ```

## Preparation

### Datasets
Please refer to [PREPARE_DATA](PREPARE_DATA.md) to prepare the datasets

### Pre-trained Weights

The pre-trained weights are stored at `config` folder, includes:
- IPBLab dataset: `config/ipblab_nof_weights.ckpt`
- Freiburg Building 079 dataset: `config/fr079_nof_weights.ckpt`
- Intel Lab dataset: `config/intel_nof_weights.ckpt`
- MIT CSAIL dataset: `config/mit_nof_weights.ckpt`

## Run Experiments

### Global Localization Experiments on IPBLab dataset
- Pre-training NOF on IPBLab dataset (**The train/eval/test set of IPBLab dataset are not available now, they will be released after our dataset paper is published!**)
  ```shell
  cd ~/ir-mcl
  bash ./shells/pretraining/ipblab.sh
  ```
- Global localization experiments
  ```shell
  cd ir-mcl
  python main.py --config_file ./config/global_localization/loc_config_{sequence_id}.yml
  # for example: python main.py --config_file ./config/global_localization/loc_config_test1.yml
  ```
  
- Pose-tracking experiments
  ```shell
  cd ir-mcl
  python main.py --config_file ./config/pose_tracking/loc_config_{sequence_id}.yml
  # for example: python main.py --config_file ./config/pose_tracking/loc_config_test1.yml
  ```

### Observation Model Experiments
- Train/Test (replace "dataset" in "fr079", "intel", or "mit")
  ```shell
  cd ir-mcl
  bash ./shells/pretraining/{dataset}.sh
  # for example: bash ./shells/pretraining/intel.sh
  ```

## Supplements for the Experimental Results
Due to the space limitation of the paper, we provide some experimental results as supplements here.

### Memery cost
We provide an ablation study on the memory cost between the occupancy grid map (OGM), Hilbert map, and our neural occupancy field (NOF). 

| Maps type             |  Approximate memory  |        Loc. method         |      RMSE: location (cm) / yaw (degree)      |
|:----------------------|:--------------------:|:--------------------------:|:--------------------------------------------:|
| OGM (5cm grid size)   |        4.00MB        |  AMCL<br>NMCL<br>SRRG-Loc  | 11.11 / 4.15<br>19.57 / 3.62<br>8.74 / 1.68  |
| OGM (10cm grid size)  |        2.00MB        |  AMCL<br>NMCL<br>SRRG-Loc  | 15.01 / 4.18<br>36.27 / 4.04<br>12.15 / 1.53 |
| Hilbert Map           |        0.01MB        |            HMCL            |                 20.04 / 4.50                 |
| NOF                   |        1.96NB        |           IR-MCL           |             **6.62** / **1.11**              |


### Ablation study on fixed particle numbers
We also provide the experiment to study the performance of global localization under the same particle numbers for all methods. We fixed the number of particles to 100,000. In the below table, all baselines and IR-MCL<sup>∗</sup> always use 100,000 particles. IR-MCL is shown for reference.

|                         Method                          |                       RMSE: location (cm) / yaw (degree)                       |
|:-------------------------------------------------------:|:------------------------------------------------------------------------------:|
| AMCL<br>NMCL<br>HMCL<br>SRRG-Loc<br>IR-MCL<sup>∗</sup>  | 11.56 / 4.12<br>19.57 / 3.62<br>20.54 / 4.70<br>8.74 / 1.68<br>6.71 / **1.11** |
|                         IR-MCL                          |                              **6.62** / **1.11**                               |

## Citation

If you use this library for any academic work, please cite our original [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/kuang2023ral.pdf).

```bibtex
@article{kuang2023ral,
  author    = {Kuang, Haofei and Chen, Xieyuanli and Guadagnino, Tiziano and Zimmerman, Nicky and Behley, Jens and Stachniss, Cyrill},
  title     = {{IR-MCL: Implicit Representation-Based Online Global Localization}},
  journal   = {IEEE Robotics and Automation Letters (RA-L)},
  doi       = {10.1109/LRA.2023.3239318},
  year      = {2023},
  codeurl   = {https://github.com/PRBonn/ir-mcl},
}
```

## Acknowledgment
This work has partially been funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement
No 101017008 (Harmony).