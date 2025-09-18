# Phase 1: Project Setup (Week 1)

### Step 1.1: Initialize Repository
```bash
# Create project structure
mkdir -p PKL-DiffusionDenoising/{pkl_dg,scripts,configs,tests,assets,docs,notebooks}
cd PKL-DiffusionDenoising

# Initialize git
git init
git add .gitignore README.md
git commit -m "Initial commit"

# Create Python package structure
touch pkl_dg/__init__.py
mkdir -p pkl_dg/{data,physics,models,guidance,baselines,evaluation,utils}
touch pkl_dg/{data,physics,models,guidance,baselines,evaluation,utils}/__init__.py
```

### Step 1.2: Setup Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0
diffusers>=0.21.0
transformers>=4.30.0
hydra-core>=1.3.0
wandb>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.21.0
kornia>=0.7.0
tifffile>=2023.0.0
cellpose>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
einops>=0.6.0
pytest>=7.3.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
jupyter>=1.0.0
ipywidgets>=8.0.0
EOF

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install package in editable mode
```

### Step 1.3: Create Package Setup
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="pkl-diffusion-denoising",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    author="Your Name",
    description="PKL-Guided Diffusion for Microscopy Denoising",
    python_requires=">=3.8",
)
```

### Step 1.4: Setup Configuration System
```yaml
# configs/config.yaml
defaults:
  - model: unet
  - data: synthesis
  - physics: microscopy
  - guidance: pkl
  - training: ddpm
  - override hydra/launcher: basic
  - _self_

experiment:
  name: ${now:%Y-%m-%d_%H-%M-%S}
  seed: 42
  device: cuda
  mixed_precision: true

paths:
  root: ${oc.env:PROJECT_ROOT,.}
  data: ${paths.root}/data
  checkpoints: ${paths.root}/checkpoints
  outputs: ${paths.root}/outputs
  logs: ${paths.root}/logs

wandb:
  project: pkl-diffusion
  entity: null  # Your W&B entity
  mode: online  # online, offline, disabled
```


