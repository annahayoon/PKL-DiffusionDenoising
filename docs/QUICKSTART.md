# Quick Start Guide

## Overview
This guide helps you get started with the PKL-Guided Diffusion system for microscopy image restoration - transforming blurry widefield images into sharp two-photon quality images using physics-aware guidance.

## Problem & Solution
- **Input**: Blurry, noisy widefield microscopy images
- **Output**: Clear, high-resolution images  
- **Method**: Diffusion model with physics guidance for Poisson noise handling

## Prerequisites

### Required Software
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM
- 100GB free disk space

## Installation

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation
```bash
# Run tests to verify setup
python -m unittest -q
```

## Quick Usage

### 1. Prepare Data
```bash
# Process real microscopy data
python scripts/preprocessing/process_microscopy_data.py \
  --wf-path /path/to/wf.tif \
  --tp-path /path/to/tp_reg.tif \
  --output-dir data/real_microscopy
```

### 2. Train Model
```bash
# Train on microscopy data
python scripts/main.py train --dataset microscopy --config configs/training/ddpm.yaml
```

### 3. Run Inference
```bash
# Denoise images
python scripts/main.py infer \
  --checkpoint checkpoints/best_model.pt \
  --input data/test/wf \
  --output outputs/denoised
```

### 4. Evaluate Results
```bash
# Compare methods
python scripts/main.py evaluate \
  --real-dir data/test/2p \
  --fake-dir outputs/denoised
```

## Key Components

- **Physics Model**: `pkl_dg/physics/` - PSF and forward model
- **Diffusion Model**: `pkl_dg/models/` - UNet and DDPM trainer
- **Guidance**: `pkl_dg/guidance/` - PKL guidance strategies
- **Data Pipeline**: `pkl_dg/data/` - Dataset and preprocessing
- **Evaluation**: `pkl_dg/evaluation/` - Metrics and robustness tests

## Configuration

The system uses Hydra for configuration management. Key config files:

- `configs/training/ddpm.yaml` - Training settings
- `configs/model/unet.yaml` - Model architecture  
- `configs/physics/microscopy.yaml` - Physics parameters
- `configs/guidance/pkl.yaml` - Guidance settings

## Common Issues

### Memory Errors
- Reduce batch size in config
- Enable mixed precision: `trainer.precision=16`
- Use CPU if needed: `trainer.accelerator=cpu`

### Poor Quality
- Train longer: increase `trainer.max_epochs`
- Check PSF alignment
- Tune guidance strength: `guidance.lambda_base`

## Next Steps

1. **Read Documentation**: See `docs/` for detailed guides
2. **Run Examples**: Check `examples/` for usage patterns
3. **Customize**: Modify configs for your data
4. **Extend**: Add new guidance strategies or models

For detailed implementation guides, see the phase-specific documentation in `docs/implementation/`.
