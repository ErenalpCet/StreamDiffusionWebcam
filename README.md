# StreamDiffusionWebcam
StreamDiffusionWebcam: A Solution for Real-Time Diffusion Generation

**Authors:** Erenalp Çetintürk

This projects code was copied from [this](https://github.com/cumulo-autumn/StreamDiffusion/blob/main/examples/screen/main.py).
This project added the ability to use the camera insted of the screen for diffusion.

We really thank for the projects owners to allow for me to use their code.

# The Projects Features
1. **Model Acceleration**
   - This project optimizes the model.
1. **Fast**
   - This project project run on RTX3050 and got 5 FPS.

# Installation
### Step 0: Clone This Repository
```bash
git clone https://github.com/ErenalpCet/StreamDiffusionWebcam.git
```
```bash
cd StreamDiffusionWebcam
```
### Step 1: Create Environment Using Anaconda
```bash
conda create -n webcamdiffusion python=3.10
```
```bash
conda activate webcamdiffusion
```
### Step 2: Install PyTorch GPU
Select the appropriate version for your system.

CUDA 11.8

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
```

CUDA 12.1

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
```

CUDA 12.4

```bash
pip3 install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu124
```
### Step 3: Install Requirements
```bash
pip install -r /path/to/requirements.txt
```
### Step 3: Customize The Prompt
In main.py line 149 and 150 please customize the prompts to your needs.

# Quick Start
```bash
python3 main.py
```
