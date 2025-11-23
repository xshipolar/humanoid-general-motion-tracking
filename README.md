<h1 align="center">General Motion Tracking for Humanoid Whole-Body Control</h1>

## Overview
This repository contains tools for motion tracking on humanoid robots using MuJoCo simulation. It is a fork adapted from the [original GMT repository](https://github.com/zixuan417/smooth-humanoid-locomotion) with extensions for scaled robot models and customizable control parameters.

## Installation && Running

First, clone this repo and install all the dependencies:
```bash
conda create -n gmt python=3.8 && conda activate gmt
pip3 install torch torchvision torchaudio
pip install "numpy==1.23.0" pydelatin tqdm opencv-python ipdb imageio[ffmpeg] mujoco mujoco-python-viewer scipy matplotlib
```
Then you can start to test the pretrained policy's performance on several example motions by running the following command:
```bash
python sim2sim.py --robot g1 --motion_file walk_stand.pkl
```
To change motions, you can replace `walk_stand.pkl` with other motions in the [motions](assets/motions/) folder.

You can also view the kinematics motion by running:
```bash
python view_motion.py --motion_file walk_stand.pkl
```

## Scaled G1 Robot Usage

This fork adds support for scaled robot models with customizable mass and control parameters. Use the `g1_scaled` robot type with various scaling options.

### Basic Scaling

**2x scaled robot with default mass/control scaling:**
```bash
python sim2sim.py --robot g1_scaled --scale 2.0 --motion_file walk_stand.pkl
```

### Advanced Scaling

**Custom mass scaling exponent (default is 3, meaning mass scales as scale³):**
```bash
python sim2sim.py --robot g1_scaled --scale 2.0 --mass_scale_alpha 2.2 --motion_file walk_stand.pkl
```

**Custom control/force scaling (for actuator limits):**
```bash
python sim2sim.py --robot g1_scaled --scale 10.0 --ctrl_scale 800.0 --motion_file walk_stand.pkl
```

**All custom parameters combined:**
```bash
python sim2sim.py --robot g1_scaled --scale 10.0 --mass_scale_alpha 2.2 --ctrl_scale 800.0 --motion_file walk_stand.pkl
```

### Command-line Arguments

- `--robot`: Robot type (`g1` or `g1_scaled`, default: `g1`)
- `--motion_file`: Motion file name in `assets/motions/` (default: `walk_stand.pkl`)
- `--scale`: Linear scale factor for `g1_scaled` (positions, sizes, mesh scale)
- `--mass_scale_alpha`: Exponent for mass scaling: `mass_scale = scale**alpha` (default: `3`)
- `--ctrl_scale`: Control/force scaling factor for `g1_scaled` (scales `ctrlrange` and `actuatorfrcrrange`)
- `--record_video`: Save simulation as MP4 video in `mujoco_videos/` directory
- `--checkpoint`: Checkpoint index (default: `-1`)

### Video Recording

Record simulations with or without scaling:
```bash
python sim2sim.py --robot g1 --motion_file walk_stand.pkl --record_video
python sim2sim.py --robot g1_scaled --scale 2.0 --motion_file walk_stand.pkl --record_video
```

## ‼️Alert & Disclaimer
Although the pretrained policy has been successfully tested on our machine, the performance of the policy might vary on different robots. We cannot guarantee the success of deployment on every machine. The model we provide is for research use only, and we disclaim all responsibility for any harm, loss, or malfunction arising from its deployment.

## News
- [ ] Data processing and retargeter code will be released soon.

## Acknowledgements
+ The Mujoco simulation script is originally adapted from [LCP](https://github.com/zixuan417/smooth-humanoid-locomotion).
+ For human motion part, we mainly refer to [ASE](https://github.com/nv-tlabs/ASE) and [PHC](https://github.com/ZhengyiLuo/PHC).

## Citation
If you find this codebase useful, please consider citing our work:
```bibtex
@article{chen2025gmt,
title={GMT: General Motion Tracking for Humanoid Whole-Body Control},
author={Chen, Zixuan and Ji, Mazeyu and Cheng, Xuxin and Peng, Xuanbin and Peng, Xue Bin and Wang, Xiaolong},
journal={arXiv:2506.14770},
year={2025}
}
```
