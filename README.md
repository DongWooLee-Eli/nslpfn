# NSL-PFN: Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks


## Overview
This is the official repository for the **[Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks (NSL-PFN)](https://arxiv.org/abs/2505.23032)** project. This repository contains code and instructions to reproduce experiments for neural scaling law extrapolation using prior-data fitted networks.

## Prerequisites
The experiments were conducted with the following setup:
- **GPU**: NVIDIA RTX A5000
- **Operating System**: Ubuntu 18.04
- **Python**: 3.7.16
- **CUDA**: 11.3
- **PyTorch**: 1.12.0

## Setup Instructions
Follow these steps to set up the environment and dependencies:

1. **Create a Conda Environment**:
   ```bash
   conda create -n nslpfn python=3.7
   conda activate nslpfn
   ```

2. **Install PyTorch**:
   ```bash
   conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
   ```

3. **Install Additional Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments
You can either use a pre-trained model or train from scratch.

### Option 1: Using Pre-trained Model
1. **Download Checkpoint**:
   ```bash
   wget https://huggingface.co/dwlee00/nslpfn/resolve/main/model.pt
   ```

2. **Run Inference**:
   ```bash
   python inference.py --checkpoint_dir ./pretrained_surrogate_results/[exp_name]
   ```

### Option 2: Training from Scratch
1. **Initialize Criterion**:
   ```bash
   python init_criterion.py --exp_name [exp_name]
   ```

2. **Train Model**:
   ```bash
   python main.py --exp_name [exp_name]
   ```

3. **Visualize and Log Results**:
   ```bash
   python inference.py --checkpoint_dir ./pretrained_surrogate_results/[exp_name]
   ```

## Notes
- Replace `[exp_name]` with your desired experiment name.
- Ensure the checkpoint directory exists and is correctly specified when running `inference.py`.