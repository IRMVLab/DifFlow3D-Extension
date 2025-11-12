# DifFlow3D: Hierarchical Diffusion Models for Uncertainty-Aware 3D Scene Flow Estimation

Jiuming Liu, Weicai Ye, Guangming Wang, Chaokang Jiang, Lei Pan, Jinru Han, Zhe Liu, Guofeng Zhang, and HeshengWang (Corresponding author)  
**TPAMI 2025**


**[Paper](https://ieeexplore.ieee.org/document/11230643)**

This repository is the official PyTorch implementation of **DifFlow3D for the 4D reconstruction task**.

## Changelog 
2025-11-12:🚀 Code on 4D reconstruction is released. More implementation details about 3D scene flow estimation can be visited at https://github.com/IRMVLab/DifFlow3D.  
2025-10-31:🎉 Our paper is accepted by TPAMI 2025.  



## Getting started
We follow LiDAR4D (CVPR2024) to establish 4D reconstruction baseline.

### 🛠️ Installation

```bash
git clone https://github.com/ispc-lab/LiDAR4D.git
cd LiDAR4D

conda create -n lidar4d python=3.9
conda activate lidar4d

# PyTorch
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA <= 11.7
# pip install torch==2.0.0 torchvision torchaudio

# Dependencies
pip install -r requirements.txt

# Local compile for tiny-cuda-nn
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch
python setup.py install

# compile packages in utils
cd utils/chamfer3D
python setup.py install

# HuangZheng@SJTU
# need to install the pointnet2 pkg pointnet2 == 0.0.0
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -e .
```


### 📁 Dataset

#### KITTI-360 dataset ([Download](https://www.cvlibs.net/datasets/kitti-360/download.php))
We use sequence00 (`2013_05_28_drive_0000_sync`) for experiments in our paper.  

<img src="https://github.com/ispc-lab/LiDAR4D/assets/51731102/c9f5d5c5-ac48-4d54-8109-9a8b745bbca0" width=65%>  

Download KITTI-360 dataset (2D images are not needed) and put them into `data/kitti360`.  
(or use symlinks: `ln -s DATA_ROOT/KITTI-360 ./data/kitti360/`).  
The folder tree is as follows:  

```bash
data
└── kitti360
    └── KITTI-360
        ├── calibration
        ├── data_3d_raw
        └── data_poses
```

Next, run KITTI-360 dataset preprocessing: (set `DATASET` and `SEQ_ID`)  

```bash
bash preprocess_data.sh
```

After preprocessing, your folder structure should look like this:  

```bash
configs
├── kitti360_{sequence_id}.txt
data
└── kitti360
    ├── KITTI-360
    │   ├── calibration
    │   ├── data_3d_raw
    │   └── data_poses
    ├── train
    ├── transforms_{sequence_id}test.json
    ├── transforms_{sequence_id}train.json
    └── transforms_{sequence_id}val.json
```

### 🚀 Run

Set corresponding sequence config path in `--config` and you can modify logging file path in `--workspace`. Remember to set available GPU ID in `CUDA_VISIBLE_DEVICES`.   
Run the following command:
```bash
# KITTI-360
bash run_kitti_lidar4d.sh
```




## Acknowledgement
We sincerely appreciate the great contribution of the following works:
- [LiDAR4D](https://github.com/ispc-lab/LiDAR4D)
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn/tree/master)
- [LiDAR-NeRF](https://github.com/tangtaogo/lidar-nerf)
- [NFL](https://research.nvidia.com/labs/toronto-ai/nfl/)
- [K-Planes](https://github.com/sarafridov/K-Planes)


## Citation
If you find our repo or paper helpful, feel free to support us with a star 🌟 or use the following citation:  
```bibtex
@ARTICLE{11230643,
  author={Liu, Jiuming and Ye, Weicai and Wang, Guangming and Jiang, Chaokang and Pan, Lei and Han, Jinru and Liu, Zhe and Zhang, Guofeng and Wang, Hesheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={DifFlow3D: Hierarchical Diffusion Models for Uncertainty-Aware 3D Scene Flow Estimation}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  keywords={Estimation;Diffusion models;Uncertainty;Three-dimensional displays;Laser radar;Dynamics;Reliability;Point cloud compression;Probabilistic logic;Noise reduction;Scene flow estimation;Diffusion model;Uncertainty evaluation;4D reconstruction;Dynamic LiDAR synthesis},
  doi={10.1109/TPAMI.2025.3629570}}
```


## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
