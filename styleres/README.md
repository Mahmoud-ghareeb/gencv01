# StyleRes - Setup and Usage Guide

This repository contains the StyleRes implementation for transforming the residuals for real image editing with StyleGAN (CVPR 2023).

## Overview

StyleRes adopts high-rate feature maps for incoming edits, ensuring the quality of the edit is not compromised while retaining most of the image details. The method supports various edits using InterfaceGAN, GANSpace, StyleClip, and GradCtrl without being explicitly trained on them.

## Quick Start

### 🖥️ Gradio Web Interface Method

The easiest way to run StyleRes is using the Gradio web interface.

#### Build and Run
```bash
cd styleres/StyleRes

# Install dependencies
conda env create -f environment.yml
conda activate styleres

# Download pretrained models to checkpoints/ directory
# StyleRes (Face): https://drive.google.com/file/d/1SXNe_txGQaGQg3AthSdwlBAlDPjlzFet/view?usp=sharing
# Facial Landmark: https://drive.google.com/file/d/1FCUAmqkVpJsNpgz4k_odYaL91gIW4hQm/view?usp=sharing

# Run the web interface
python app.py --device=cuda
```

#### Access the Web Interface
- Open your browser and go to: `http://127.0.0.1:7860`
- The Gradio interface will be available with:
  - **Method Selection**: Choose from InterfaceGAN, GANSpace, StyleClip, or GradCtrl
  - **Edit Selection**: Pick specific edits based on the chosen method
  - **Strength Control**: Adjust editing power
  - **Face Alignment**: Optional crop and align functionality

### 🖥️ CLI Method

#### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/hamzapehlivan/StyleRes.git
cd StyleRes
```

2. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate styleres
```

3. **Download pretrained models:**
Download the required models and place them in the `checkpoints/` directory:
- StyleRes (Face): [Download Link](https://drive.google.com/file/d/1SXNe_txGQaGQg3AthSdwlBAlDPjlzFet/view?usp=sharing)
- Facial Landmark: [Download Link](https://drive.google.com/file/d/1FCUAmqkVpJsNpgz4k_odYaL91gIW4hQm/view?usp=sharing)

4. **Install additional dependencies for alignment (optional):**
```bash
apt install cmake
pip install dlib scipy
```

5. **Set environment variables and run:**

in **run_cli** you can adjust the parameters and run the following

```bash
bash run_cli.sh
```

## Features

### InterfaceGAN Editing
- **Smile**: Adjust facial expression
- **Age**: Modify apparent age
- **Pose**: Change head orientation

### GANSpace Editing (42 different edits)
- **Hair**: White Hair, Curly Hair, Dark Hair, Frizzy Hair, Straight Bowl Cut
- **Facial Expression**: Big Smile, Huge Grin, Wide Smile, Caricature Smile, Screaming
- **Facial Features**: Eye Openness, Eyebrow Thickness, Nose Length, Large Jaw
- **Appearance**: Lipstick, Mascara, Trimmed Beard, Wrinkles, Eye Wrinkles
- **Background**: Background Blur, Overexposed, Sunlight In Face

### StyleClip Editing
- **Mapper Network**: 9 predefined edits
- **Global Directions**: 17 example directions with custom text prompts
- **CLIP Integration**: Natural language-based editing

### GradCtrl Editing
- **Smile**: Expression modification
- **Age**: Age transformation
- **Eyeglasses**: Add/remove glasses
- **Gender**: Gender transformation

### Advanced Options
- **Face Alignment**: Automatic face detection and cropping
- **Device Selection**: CPU or CUDA execution
- **Strength Control**: Adjustable editing power
- **Batch Processing**: Process multiple images via inference scripts

## CLI Usage Examples

**List available methods:**
```bash
python cli.py --list-methods
```

**List available edits for a method:**
```bash
python cli.py --list-edits GANSpace
```

**Edit a single image:**
```bash
python cli.py \
    --input-image path/to/image.jpg \
    --output-image path/to/edited.jpg \
    --method GANSpace \
    --edit "White Hair" \
    --factor -5.0 \
    --align
```

**Batch processing:**
```bash
python inference.py \
    --datadir=samples/inference_samples \
    --outdir=results \
    --edit_configs=options/editing_options/template.py
```
