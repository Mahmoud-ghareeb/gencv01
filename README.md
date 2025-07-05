# Face Manipulation Methods Repository

This repository contains implementations of advanced face manipulation and image editing methods using deep learning techniques.

## Overview

The repository includes two state-of-the-art methods for face manipulation and style transfer:

1. **SFE (StyleFeatureEditor)** - Detail-rich StyleGAN inversion and high-quality image editing
2. **StyleRes** - Real image editing by transforming the residuals with StyleGAN

## Repository Structure

```
├── SFE/                          # StyleFeatureEditor implementation
│   ├── StyleFeatureEditor/       # Main code directory
│   ├── images/                   # Image assets
│   │   ├── base_images/          # Input images
│   │   └── edited_images/        # Generated output images
│   └── README.md                 # Detailed SFE documentation
└── styleres/                     # StyleRes implementation
    ├── StyleRes/                 # Main code directory
    ├── images/                   # Image assets
    │   ├── base_images/          # Input images
    │   └── edited_images/        # Generated output images
    └── README.md                 # Detailed StyleRes documentation

```

## Methods

### 1. StyleFeatureEditor (SFE)
- **Purpose**: Detail-rich StyleGAN inversion and high-quality image editing
- **Features**: 
  - Standard editing with predefined directions (age, glasses, smiling, etc.)
  - Custom StyleCLIP text-based editing
  - Docker-based deployment with web interface
- **Usage**: See `SFE/README.md` for detailed instructions

### 2. StyleRes
- **Purpose**: Real image editing by transforming StyleGAN residuals
- **Features**:
  - InterfaceGAN editing (smile, age, pose)
  - GANSpace editing (42 different transformations)
  - StyleClip integration
  - GradCtrl editing methods
- **Usage**: See `styleres/README.md` for detailed instructions

## Quick Start

Each method has its own comprehensive documentation:

- **SFE**: Navigate to `SFE/README.md` for Docker setup and usage instructions
- **StyleRes**: Navigate to `styleres/README.md` for conda environment setup and CLI usage
