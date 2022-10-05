# IR-MCL: Implicit Representation-Based Online Global Localization
This repo contains the code of the paper:
*IR-MCL: Implicit Representation-Based Online Global Localization*,

by [Haofei Kuang](https://www.ipb.uni-bonn.de/people/haofei-kuang/), [Xieyuanli Chen](https://www.ipb.uni-bonn.de/people/xieyuanli-chen/), [Tiziano Guadagnino](https://phd.uniroma1.it/web/TIZIANO-GUADAGNINO_nP1536210_IT.aspx), [Nicky Zimmerman](https://www.ipb.uni-bonn.de/people/nicky-zimmerman/), [Jens Behley](https://www.ipb.uni-bonn.de/people/jens-behley/) and [Cyrill Stachniss](https://www.ipb.uni-bonn.de/people/cyrill-stachniss/)

----
### IR-MCL pipeline.
<p align="center">
<img src="https://user-images.githubusercontent.com/18661888/194111809-4f966ab5-64be-45fc-963b-a6fe0a8c14ed.png" width="700"/>
</p>

### Online localization demo
<p align="center">
<img src="https://user-images.githubusercontent.com/18661888/194112420-f83c2d02-e33b-4e8f-87df-bcaab12641a2.gif" width="800">
</p>


## Abstract
Determining the state of a mobile robot is an essential building block of robot navigation systems. In this paper, we address the problem of estimating the robot’s pose in an indoor environment using 2D LiDAR data and investigate how modern environment models can improve gold standard Monte-Carlo localization (MCL) systems. We propose a neural occupancy field (NOF) to implicitly represent the scene using a neural network. With the pretrained network, we can synthesize 2D LiDAR scans for an arbitrary robot pose through volume rendering. Based on the implicit representation, we can obtain the similarity between a synthesized and actual scan as an observation model and integrate it into an MCL system to perform accurate localization. We evaluate our approach on five sequences of a self-recorded dataset and three publicly available datasets. We show that we can accurately and efficiently localize a robot using our approach surpassing the localization performance of state-of-the-art methods. The experiments suggest that the presented implicit representation is able to predict more accurate 2D LiDAR scans leading to an improved observation model for our particle filter-based localization.

## Code
Coming soon!

## Acknowledgment
This work has partially been funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement
No 101017008 (Harmony).
