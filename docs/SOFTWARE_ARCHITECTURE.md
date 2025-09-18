# Software Architecture for PKL-Guided Diffusion

## Overview
This document describes the production-ready software architecture for implementing **Poisson-Kullback-Leibler (PKL) Guided Diffusion** for microscopy image restoration, as described in the ICLR 2025 paper "Microscopy Denoising Diffusion with Poisson-aware Physical Guidance".

## System Architecture

### Core Components

```
PKL-DiffusionDenoising/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── model/                 # Model configurations
│   │   ├── unet.yaml
│   │   └── diffusion.yaml
│   ├── data/                  # Data configurations
│   │   ├── synthesis.yaml
│   │   └── microscopy.yaml
│   ├── physics/               # Physical model configs
│   │   ├── psf.yaml
│   │   └── noise.yaml
│   ├── guidance/              # Guidance mechanism configs
│   │   ├── l2.yaml
│   │   ├── anscombe.yaml
│   │   └── pkl.yaml
│   ├── training/              # Training configs
│   └── evaluation/            # Evaluation configs
│
├── pkl_dg/                    # Main package
│   ├── __init__.py
│   ├── data/                  # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py        # PyTorch datasets
│   │   ├── synthesis.py      # Training data synthesis
│   │   ├── transforms.py     # Image transforms
│   │   └── dataloader.py     # Custom data loaders
│   │
│   ├── physics/               # Physical modeling
│   │   ├── __init__.py
│   │   ├── forward_model.py  # WF→2P forward operator
│   │   ├── psf.py           # PSF handling
│   │   ├── noise.py         # Noise models (Poisson, Gaussian)
│   │   └── operators.py     # Linear operators (A, A^T)
│   │
│   ├── models/               # Neural network models
│   │   ├── __init__.py
│   │   ├── unet.py          # UNet wrapper (diffusers)
│   │   ├── diffusion.py    # DDPM training
│   │   ├── sampler.py      # DDIM sampling
│   │   └── ema.py          # EMA weights
│   │
│   ├── guidance/            # Guidance mechanisms
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract guidance class
│   │   ├── l2.py           # L2 guidance
│   │   ├── anscombe.py     # Anscombe + L2
│   │   ├── pkl.py          # PKL guidance (ours)
│   │   └── schedules.py    # Adaptive λ_t scheduling
│   │
│   ├── baselines/           # Baseline methods
│   │   ├── __init__.py
│   │   ├── richardson_lucy.py
│   │   └── rcan.py
│   │
│   ├── evaluation/          # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py      # PSNR, SSIM, FRC
│   │   ├── cellpose_eval.py # Downstream task eval
│   │   ├── robustness.py   # PSF mismatch, alignment
│   │   └── hallucination.py # Adversarial tests
│   │
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── io.py           # File I/O
│       ├── normalization.py # Scale conversions
│       ├── visualization.py # Plotting
│       └── logging.py      # W&B integration
│
├── scripts/                 # Executable scripts
│   ├── train_diffusion.py  # Train unconditional DDPM
│   ├── synthesize_data.py  # Generate training data
│   ├── inference.py        # Run guided inference
│   ├── evaluate.py         # Run evaluation suite
│   └── download_data.py    # Download datasets
│
├── notebooks/               # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── psf_analysis.ipynb
│   └── results_visualization.ipynb
│
├── tests/                   # Unit tests
│   ├── test_physics.py
│   ├── test_guidance.py
│   ├── test_metrics.py
│   └── test_data.py
│
├── assets/                  # Static assets
│   ├── psf/                # Measured PSFs
│   └── examples/           # Example images
│
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
└── README.md               # Project documentation
```

## Technology Stack

### Core Dependencies
- **PyTorch** (2.0+): Deep learning framework
- **PyTorch Lightning** (2.0+): Training orchestration
- **Hugging Face Diffusers** (0.21+): Pre-built diffusion components
- **Hydra** (1.3+): Configuration management
- **Weights & Biases**: Experiment tracking

### Scientific Computing
- **NumPy** (1.24+): Numerical operations
- **SciPy** (1.10+): Scientific algorithms
- **scikit-image** (0.21+): Image processing, metrics
- **kornia** (0.7+): Differentiable image operations
- **tifffile**: Microscopy image I/O

### Specialized Tools
- **Cellpose** (2.0+): Cell segmentation for evaluation
- **matplotlib/seaborn**: Visualization
- **tqdm**: Progress bars
- **einops**: Tensor operations

## Data Flow Architecture

### Training Pipeline
```
External Images
        ↓
[Data Synthesis Module]
    - Load clean images
    - Apply PSF convolution
    - Add Poisson noise
    - Add Gaussian background
        ↓
[Training Dataset]
    - 2P-like targets (x)
    - WF-like inputs (y) [stored but not used in unconditional training]
        ↓
[DDPM Training]
    - Unconditional diffusion on x
    - Noise prediction network ε_θ
        ↓
[Trained Model Checkpoint]
```

### Inference Pipeline
```
WF Measurement (y)
        ↓
[Initialization]
    - Sample x_T ~ N(0, I)
        ↓
[DDIM Sampling Loop] (t = T → 1)
    - Predict x̂_0 from x_t
    - Denormalize to intensity
    - Apply guidance gradient
    - Adaptive schedule λ_t
    - Update x̂_0 → x̂'_0
    - DDIM step → x_{t-1}
        ↓
[Reconstructed 2P Image]
```

## Key Design Patterns

### 1. Strategy Pattern for Guidance
```python
class GuidanceStrategy(ABC):
    @abstractmethod
    def compute_gradient(self, x0_hat, y, forward_op, background):
        pass

class PKLGuidance(GuidanceStrategy):
    def compute_gradient(self, x0_hat, y, forward_op, background):
        Ax = forward_op(x0_hat) + background
        return forward_op.adjoint(1.0 - y / (Ax + self.eps))
```

### 2. Factory Pattern for Model Creation
```python
class ModelFactory:
    @staticmethod
    def create_diffusion_model(config):
        unet = UNet2DModel.from_config(config.model)
        return DiffusionModel(unet, config.diffusion)
    
    @staticmethod
    def create_guidance(guidance_type, config):
        strategies = {
            'l2': L2Guidance,
            'anscombe': AnscombeGuidance,
            'pkl': PKLGuidance
        }
        return strategies[guidance_type](config)
```

### 3. Builder Pattern for Forward Model
```python
class ForwardModelBuilder:
    def __init__(self):
        self.psf = None
        self.background = 0
        self.noise_model = None
    
    def with_psf(self, psf_path):
        self.psf = load_psf(psf_path)
        return self
    
    def with_background(self, background):
        self.background = background
        return self
    
    def with_poisson_noise(self):
        self.noise_model = PoissonNoise()
        return self
    
    def build(self):
        return ForwardModel(self.psf, self.background, self.noise_model)
```

## Configuration Management

### Hydra Configuration Structure
```yaml
# config.yaml
defaults:
  - model: unet
  - data: microscopy
  - physics: widefield
  - guidance: pkl
  - training: ddpm
  - _self_

experiment:
  name: pkl_guidance_experiment
  seed: 42

paths:
  data_dir: ${oc.env:DATA_DIR,data/}
  checkpoint_dir: ${oc.env:CHECKPOINT_DIR,checkpoints/}
  output_dir: ${oc.env:OUTPUT_DIR,outputs/}
```

### Model Configuration
```yaml
# model/unet.yaml
# Passed directly to diffusers.UNet2DModel constructor
# See https://huggingface.co/docs/diffusers/api/models/unet2d for all options
_target_: diffusers.UNet2DModel
sample_size: 256
in_channels: 1
out_channels: 1
layers_per_block: 2
block_out_channels: [128, 128, 256, 512, 512]
down_block_types:
  - "DownBlock2D"
  - "DownBlock2D"
  - "AttnDownBlock2D"
  - "AttnDownBlock2D"
  - "DownBlock2D"
up_block_types:
  - "UpBlock2D"
  - "AttnUpBlock2D"
  - "AttnUpBlock2D"
  - "UpBlock2D"
  - "UpBlock2D"
attention_head_dim: 8
```

### Guidance Configuration
```yaml
# guidance/pkl.yaml
type: pkl
lambda_base: 0.1
epsilon: 1e-6
schedule:
  type: adaptive
  T_threshold: 800
  epsilon_lambda: 1e-3
```

## Component Interface Specification
This section defines the expected inputs, outputs, and core responsibilities for each major component. This serves as a contract to guide implementation and unit testing.

### 1. Physics (`pkl_dg/physics`)
- **Component**: `ForwardModel`
- **Responsibility**: To apply the physical forward operator \\( \mathcal{A}(\cdot) \\) and its adjoint \\( \mathcal{A}^T(\cdot) \\), representing the microscopy imaging process.
- **Key Methods**:
    - `__init__(self, psf: torch.Tensor, background: float, device: str)`
    - `apply_psf(self, x: torch.Tensor) -> torch.Tensor`
        - **Input `x`**: `torch.Tensor` of shape `[B, C, H, W]`, representing the clean image in the intensity domain.
        - **Output**: `torch.Tensor` of shape `[B, C, H, W]`, representing the blurred image \\( \mathcal{A}(\mathbf{x}) \\).
    - `apply_psf_adjoint(self, y: torch.Tensor) -> torch.Tensor`
        - **Input `y`**: `torch.Tensor` of shape `[B, C, H, W]`.
        - **Output**: `torch.Tensor` of shape `[B, C, H, W]`, representing the result of the adjoint operation \\( \mathcal{A}^T(\mathbf{y}) \\).
    - `forward(self, x: torch.Tensor, add_noise: bool) -> torch.Tensor`
        - **Input `x`**: `torch.Tensor` of shape `[B, C, H, W]`, clean image.
        - **Input `add_noise`**: `bool`, flag to apply Poisson noise.
        - **Output**: `torch.Tensor` of shape `[B, C, H, W]`, representing the simulated measurement \\( \mathbf{y} \\).

### 2. Data (`pkl_dg/data`)
- **Component**: `IntensityToModel`
- **Responsibility**: To perform the reversible transformation between the physical intensity domain (`[0, max_intensity]`) and the normalized model domain (`[-1, 1]`).
- **Key Methods**:
    - `__call__(self, x: torch.Tensor) -> torch.Tensor` (intensity to model)
        - **Input `x`**: `torch.Tensor` (any shape), in intensity domain.
        - **Output**: `torch.Tensor` (same shape), in model domain `[-1, 1]`.
    - `inverse(self, x: torch.Tensor) -> torch.Tensor` (model to intensity)
        - **Input `x`**: `torch.Tensor` (any shape), in model domain `[-1, 1]`.
        - **Output**: `torch.Tensor` (same shape), in non-negative intensity domain.

### 3. Model (`pkl_dg/models`)
- **Component**: `DenoisingUNet`
- **Responsibility**: To act as the core noise prediction network \\( \boldsymbol{\epsilon}_\theta \\).
- **Key Methods**:
    - `forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor`
        - **Input `x`**: `torch.Tensor` of shape `[B, C, H, W]`, the noisy image \\( \mathbf{x}_t \\) in the model domain.
        - **Input `t`**: `torch.Tensor` of shape `[B]`, the timesteps.
        - **Output**: `torch.Tensor` of shape `[B, C, H, W]`, the predicted noise \\( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \\).

### 4. Guidance (`pkl_dg/guidance`)
- **Component**: `GuidanceStrategy` (Abstract Base Class)
- **Responsibility**: To define the interface for all guidance mechanisms.
- **Key Methods**:
    - `compute_gradient(self, x0_hat: torch.Tensor, y: torch.Tensor, forward_model: 'ForwardModel', t: int) -> torch.Tensor`
        - **Input `x0_hat`**: `torch.Tensor` of shape `[B, C, H, W]`, the predicted clean image in the **intensity domain**.
        - **Input `y`**: `torch.Tensor` of shape `[B, C, H, W]`, the physical measurement.
        - **Input `forward_model`**: `ForwardModel` instance.
        - **Input `t`**: `int`, the current timestep.
        - **Output**: `torch.Tensor` of shape `[B, C, H, W]`, the computed guidance gradient.

### 5. Sampler (`pkl_dg/models`)
- **Component**: `DDIMSampler`
- **Responsibility**: To orchestrate the entire reverse diffusion process, incorporating guidance to generate a clean image from a measurement.
- **Key Methods**:
    - `sample(self, y: torch.Tensor, shape: tuple, device: str) -> torch.Tensor`
        - **Input `y`**: `torch.Tensor` of shape `[B, C, H, W]`, the physical measurement in the intensity domain.
        - **Input `shape`**: `tuple`, the desired output shape, e.g., `(1, 1, 256, 256)`.
        - **Input `device`**: `str`, the target computation device.
        - **Output**: `torch.Tensor` of shape `[B, C, H, W]`, the final reconstructed image in the **intensity domain**.

## Performance Optimizations

### 1. FFT-based Convolution
- Use FFT for PSF convolution (O(n log n) vs O(n²))
- Cache FFT of PSF for repeated use

### 2. Mixed Precision Training
- Use automatic mixed precision (AMP) for training
- FP16 for forward pass, FP32 for gradients

### 3. Gradient Checkpointing
- Enable for large UNet models to reduce memory usage

### 4. Parallel Data Loading
- Multi-worker DataLoader with prefetching
- Pin memory for GPU transfer

### 5. Compiled Models (PyTorch 2.0)
```python
model = torch.compile(model, mode="reduce-overhead")
```

## Testing Strategy

### Unit Tests
- Physics operators (PSF convolution, adjoint)
- Noise models (Poisson, Gaussian)
- Guidance gradients (numerical gradient checking)
- Normalization/denormalization

### Integration Tests
- End-to-end training pipeline
- Inference with different guidance methods
- Metric computation on synthetic data

### System Tests
- Full evaluation suite on test dataset
- Robustness tests (PSF mismatch, alignment)
- Hallucination tests

## Deployment Considerations

### Model Serving
- Export to ONNX for inference optimization
- TorchScript for production deployment
- Optional: TensorRT for NVIDIA GPU acceleration

### Resource Requirements
- **Training**: 1-4 GPUs (A100/V100 recommended)
- **Inference**: 1 GPU (can run on consumer GPUs)
- **Memory**: 16-32GB GPU memory for training
- **Storage**: ~100GB for datasets and checkpoints

### Monitoring
- Weights & Biases for experiment tracking
- TensorBoard for local monitoring
- Custom metrics dashboard for production

## Security & Best Practices

### Code Quality
- Type hints throughout codebase
- Docstrings for all public methods
- Pre-commit hooks (black, isort, flake8)

### Reproducibility
- Fixed random seeds
- Version pinning in requirements
- Config versioning with git

### Data Management
- Checksums for downloaded data
- Data versioning with DVC (optional)
- Separate train/val/test splits

## Extension Points

### Adding New Guidance Methods
1. Inherit from `GuidanceStrategy`
2. Implement `compute_gradient` method
3. Register in factory
4. Add configuration file

### Adding New Baselines
1. Create module in `baselines/`
2. Implement common interface
3. Add to evaluation suite

### Custom Metrics
1. Add to `evaluation/metrics.py`
2. Register in evaluation config
3. Update visualization tools

## References
- Paper: "Microscopy Denoising Diffusion with Poisson-aware Physical Guidance"
- Diffusers Documentation: https://huggingface.co/docs/diffusers
- PyTorch Lightning: https://lightning.ai/docs/pytorch
- Hydra: https://hydra.cc/docs/intro/
