# StyleFeatureEditor - Setup and Usage Guide

This repository contains the StyleFeatureEditor implementation for detail-rich StyleGAN inversion and high-quality image editing.

## Overview

StyleFeatureEditor enables editing in both w-latents and F-latents, allowing for reconstruction of finer image details while preserving them during editing. The method excels in reconstruction quality and can edit challenging out-of-domain examples.

## Quick Start

### 🐳 Docker Method (Recommended)

The easiest way to run StyleFeatureEditor is using Docker with GPU support.

#### Prerequisites
- Docker with GPU support (nvidia-docker)
- NVIDIA GPU with CUDA support
- At least 8GB GPU memory

#### Build and Run
```bash
cd SFE/StyleFeatureEditor

docker compose up --build
```

#### Access the Web Interface
- Open your browser and go to: `http://localhost:7860`
- The Gradio interface will be available with two tabs:
  - **Standard Editing**: Choose from predefined editing types (age, glasses, smiling, etc.)
  - **Custom StyleCLIP**: Text-based editing with custom prompts

### 🖥️ CLI Method

For direct CLI usage or development, follow these steps:

#### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- CMAKE
- Python 3.10
- Conda

#### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/ControlGenAI/StyleFeatureEditor
cd StyleFeatureEditor
```

2. **Create and activate conda environment:**
```bash
conda create -n sfe python=3.10 -y
conda activate sfe
```

3. **Install dependencies:**
```bash
bash env_install.sh
```

4. **Download pretrained models:**
```bash
git clone https://huggingface.co/AIRI-Institute/StyleFeatureEditor
cd StyleFeatureEditor && git lfs pull && cd ..
mv StyleFeatureEditor/pretrained_models pretrained_models
rm -rf StyleFeatureEditor
```

5. **Set environment variables and run:**

in **run_cli** you can adjust the parameters and run the following

```bash
export CC=gcc-9
export CXX=g++-9
bash run_cli.sh
```

## Features

### Standard Editing
- **Predefined directions**: Age, glasses, smiling, hair styles, etc.
- **Power control**: Adjust editing strength from -15 to +15
- **Face alignment**: Automatic face detection and alignment
- **Masking**: Preserve background while editing face regions

### StyleCLIP Editing
- **Text-based editing**: Use natural language prompts
- **Custom transformations**: "face" → "face with curly afro"
- **Disentanglement control**: Fine-tune editing precision
- **Flexible prompting**: Any text description supported

### Advanced Options
- **Mask threshold**: Control background preservation (0.01-0.3)
- **Alignment**: Automatic face cropping and resizing
- **Batch processing**: Process multiple images via dataset scripts
- **Unalignment**: Restore edited faces to original image context
