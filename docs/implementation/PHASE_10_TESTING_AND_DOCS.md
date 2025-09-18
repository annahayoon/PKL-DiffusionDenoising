# Phase 10: Testing and Documentation (Week 6)

### Step 10.1: Create Unit Tests
```python
# tests/test_physics.py
import pytest
import torch
import numpy as np
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel

def test_psf_normalization():
    """Test PSF normalization."""
    psf = PSF()
    assert np.abs(psf.psf.sum() - 1.0) < 1e-6

def test_forward_model_shape():
    """Test forward model preserves shape."""
    psf = PSF()
    forward_model = ForwardModel(
        psf=psf.to_torch(),
        background=0.0,
        device='cpu'
    )
    
    x = torch.randn(2, 1, 256, 256)
    y = forward_model.forward(x)
    
    assert y.shape == x.shape

def test_adjoint_operator():
    """Test adjoint is correct."""
    psf = PSF()
    forward_model = ForwardModel(
        psf=psf.to_torch(),
        background=0.0,
        device='cpu'
    )
    
    # Test <Ax, y> = <x, A^T y>
    x = torch.randn(1, 1, 64, 64)
    y = torch.randn(1, 1, 64, 64)
    
    Ax = forward_model.apply_psf(x)
    ATy = forward_model.apply_psf_adjoint(y)
    
    inner1 = (Ax * y).sum()
    inner2 = (x * ATy).sum()
    
    assert torch.abs(inner1 - inner2) / torch.abs(inner1) < 1e-5
```

### Step 10.2: Create README
```markdown
# PKL-Diffusion Denoising

Implementation of "Microscopy Denoising Diffusion with Poisson-aware Physical Guidance" (ICLR 2025).

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/PKL-DiffusionDenoising.git
cd PKL-DiffusionDenoising

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Prepare Data

```bash
# Download ImageNet subset
python scripts/download_data.py --data-dir data/

# Synthesize training data
python scripts/synthesize_data.py \
    --source-dir data/imagenet \
    --output-dir data/synthesized \
    --psf assets/psf/measured_psf.tif
```

### 2. Train Model

```bash
python scripts/train_diffusion.py \
    training.batch_size=32 \
    training.max_epochs=200 \
    wandb.mode=online
```

### 3. Run Inference

```bash
# PKL guidance (recommended)
python scripts/inference.py \
    guidance=pkl \
    inference.checkpoint_path=checkpoints/best_model.pt \
    inference.input_dir=data/test/wf \
    inference.output_dir=outputs/pkl

# L2 guidance (baseline)
python scripts/inference.py \
    guidance=l2 \
    inference.checkpoint_path=checkpoints/best_model.pt \
    inference.input_dir=data/test/wf \
    inference.output_dir=outputs/l2
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --pred-dir outputs/pkl \
    --target-dir data/test/2p \
    --metrics psnr,ssim,frc
```

## Configuration

All configurations are managed through Hydra. Key config files:

- `configs/config.yaml`: Main configuration
- `configs/model/unet.yaml`: UNet architecture
- `configs/guidance/pkl.yaml`: PKL guidance settings
- `configs/training/ddpm.yaml`: Training hyperparameters

## Citation

```bibtex
@inproceedings{anonymous2025pkl,
  title={Microscopy Denoising Diffusion with Poisson-aware Physical Guidance},
  author={Anonymous},
  booktitle={ICLR},
  year={2025}
}
```

## License

MIT License
```
