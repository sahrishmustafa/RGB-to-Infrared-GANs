# RGB-to-Infrared Image Translation using GANs

A deep learning project that explores visible-to-infrared (RGB → thermal) image translation using three Generative Adversarial Network architectures: **Pix2Pix**, **InfraGAN**, and **ClawGAN**. The project includes full training notebooks, a modular inference application, and Docker support for deployment.

> **Course Project** — Generative AI  

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Models](#models)
- [Dataset](#dataset)
- [Training](#training)
- [Inference App](#inference-app)
- [Docker Deployment](#docker-deployment)
- [Requirements](#requirements)
- [Results](#results)
- [Prompts Log](#prompts-log)

---

## Overview

This project investigates cross-modal image translation from visible (RGB) to infrared (thermal) imagery. Three GAN architectures are implemented, trained, and evaluated on the [KAIST Multispectral Pedestrian Dataset](https://sites.google.com/site/pedestrianbenchmark/), with quantitative evaluation using PSNR, SSIM, and LPIPS metrics.

The core idea: given a daytime RGB image, synthesize what that scene would look like in the infrared spectrum — a task with applications in surveillance, autonomous driving, and night-vision systems.

---

## Repository Structure

```
.
├── src/                          # Training source code
│   ├── base_model/
│   │   └── pix2pix.py            # Pix2Pix GAN (training script)
│   ├── clawGAN/
│   │   └── clawgan.ipynb         # ClawGAN training notebook
│   └── infraGAN/
│       └── infragan.ipynb        # InfraGAN training notebook
│
├── infrared-inference-app/       # Streamlit inference web app
│   ├── app.py                    # Main Streamlit application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── basic.py              # Basic U-Net generator (Pix2Pix-style)
│   │   ├── clawgan.py            # ClawGAN dual-decoder generator
│   │   └── infragan.py           # InfraGAN U-Net generator
│   ├── checkpoints/              # Place trained .pth model weights here
│   ├── requirements.txt          # App-specific dependencies
│   └── Dockerfile                # Docker config for the inference app
│
├── preprocess/
│   └── iris_preprocessing.py     # Preprocessing utilities for the IRIS face dataset
│
├── reports/
│   ├── Project_Proposal.pdf
│   └── Report.pdf
│
└── requirements.txt              # Top-level training dependencies
```

---

## Models

### 1. Pix2Pix (`src/base_model/pix2pix.py`)

The baseline model. A supervised image-to-image translation framework using:
- **Generator**: 5-level U-Net (encoder-decoder with skip connections), mapping 3-channel RGB → 1-channel IR
- **Discriminator**: PatchGAN — predicts authenticity at a patch level rather than globally
- **Loss**: Combined cGAN adversarial loss + L1 pixel loss (λ=100) + SSIM loss (λ=5.0)
- **Stabilization**: Soft/flipped labels and label smoothing (real labels set to 0.9) to prevent discriminator dominance

Key hyperparameters: `IMG_SIZE=256`, `BATCH_SIZE=6`, `LR=2e-4`, `EPOCHS=20`, `MAX_SAMPLES=20000`

---

### 2. ClawGAN (`src/clawGAN/clawgan.ipynb` · `infrared-inference-app/models/clawgan.py`)

A CycleGAN-inspired architecture with a novel dual-decoder "Claw" generator:
- **Generator**: Shared encoder → two parallel U-Net++ style decoders (Decoder A and Decoder B). Their outputs are averaged before the final convolution
- Uses **InstanceNorm** (instead of BatchNorm) to prevent batch statistics from interfering with image translation — this was a key finding from ablation studies
- **Loss**: Adversarial + Synthetic (L_syn) + Cycle consistency (L_cyc, λ=15.0) + Identity (L_id) + Feature Reconstruction (L_fr, via VGG-16)
- Three ablation experiments were run, with Experiment 3 (KAIST dataset + InstanceNorm + λ_cyc=15.0) producing the best results

Architecture: `in_channels=3 → base_filters=32 → out_channels=1`

---

### 3. InfraGAN (`src/infraGAN/infragan.ipynb` · `infrared-inference-app/models/infragan.py`)

An architecture designed for high structural fidelity in thermal synthesis:
- **Generator**: Recursive U-Net with 4 downsampling levels (3→64→128→256→512)
- **Dual Discriminators**: An Encoder-Decoder discriminator + a Pixel-level discriminator — enforcing both global coherence and per-pixel realism
- **Loss**: Adversarial + L1 (λ=100) + SSIM (λ=100) — the high SSIM weight prioritizes structural preservation
- Memory-optimized to run on a 6 GB GPU (reduced filter sizes, 3 levels instead of 4 where necessary)

Key hyperparameters: `IMG_SIZE=256`, `BATCH_SIZE=2`, `LR_G=2e-4`, `LR_D=LR_G/100`, `EPOCHS=20`

---

## Dataset

**[KAIST Multispectral Pedestrian Dataset](https://sites.google.com/site/pedestrianbenchmark/)**

- Paired RGB + Long-Wave Infrared (LWIR/thermal) images at 640×480 resolution
- Training uses a subset of **20,000 paired images**
- Preprocessing: center-crop → resize to 256×256; RGB normalized to [-1, 1] (3 channels); thermal converted to grayscale and normalized to [-1, 1] (1 channel)
- Dataset directory structure expected:
  ```
  kaist-dataset/
  └── setXX/
      └── VXXX/
          ├── visible/   ← RGB frames
          └── lwir/      ← Thermal frames
  ```

A secondary dataset, the **IRIS face dataset**, was used for ClawGAN ablation experiments. A template-matching alignment function (`preprocess/iris_preprocessing.py`) handles the slight sensor misalignment between its visible and thermal cameras.

---

## Training

### Pix2Pix

Run directly as a Python script:

```bash
cd src/base_model
python pix2pix.py
```

Update `DATASET_ROOT` and `CHECKPOINT_DIR` inside the script before running.

### ClawGAN & InfraGAN

Open and run the respective Jupyter notebooks:

```bash
jupyter notebook src/clawGAN/clawgan.ipynb
jupyter notebook src/infraGAN/infragan.ipynb
```

Update the `DATASET_ROOT` and checkpoint paths in the hyperparameter cells at the top of each notebook.

**Hardware**: All models were trained on a single ~6 GB GPU. Training takes approximately 20 epochs.

---

## Inference App

A Streamlit web application that lets you upload an RGB image and run inference with any of the three trained models.

### Setup

1. Install dependencies:
   ```bash
   cd infrared-inference-app
   pip install -r requirements.txt
   ```

2. Place your trained model checkpoints in `infrared-inference-app/checkpoints/`:
   ```
   checkpoints/
   ├── clawgan.pth
   ├── infragan.pth
   └── basic.pth
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`, select a model from the sidebar, upload an RGB image, and click **Convert to Infrared**.

> **Note**: If a checkpoint is not found, the app will run with random (untrained) weights and display a warning. This is useful for testing the interface.

---

## Docker Deployment

The inference app is fully containerized. The Docker image uses `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` as its base, so CUDA is available if the host has an NVIDIA GPU.

### Build

```bash
cd infrared-inference-app
docker build -t rgb-to-ir-app .
```

### Run

```bash
docker run -p 8501:8501 rgb-to-ir-app
```

Then visit `http://localhost:8501`.

To mount external checkpoints without rebuilding:

```bash
docker run -p 8501:8501 -v /path/to/your/checkpoints:/app/checkpoints rgb-to-ir-app
```

---

## Requirements

### Training (top-level)

```
torch, torchvision
datasets, scikit-learn, scikit-image
matplotlib, pandas, tqdm, numpy, Pillow
transformers, diffusers, accelerate
einops, timm, torchmetrics
lpips, pytorch-msssim, torch-fidelity
opencv-python
```

Install with:
```bash
pip install -r requirements.txt
```

### Inference App

```
numpy
pillow
streamlit
```

PyTorch is already included in the Docker base image. If running locally without Docker, install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/) before installing the app requirements.

---

## Results

Models are evaluated on a held-out validation split using three metrics:

| Metric | Description |
|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio — higher is better |
| **SSIM** | Structural Similarity Index — higher is better (max 1.0) |
| **LPIPS** | Learned Perceptual Image Patch Similarity — lower is better |

Refer to the full quantitative comparison table in [`reports/Report.pdf`](reports/Report.pdf).

## License

This project was developed for academic purposes. The KAIST dataset has its own terms of use — please refer to the [official dataset page](https://sites.google.com/site/pedestrianbenchmark/) before using it.
