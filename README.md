PKL-DiffusionDenoising
======================

PKL-Guided Diffusion for Microscopy Denoising.

This repository follows a phased implementation plan. Phase 1 sets up the
project structure, environment, and configuration system.

Data policy
-----------

- Data, logs, weights, and large artifacts are intentionally excluded from version control via `.gitignore`.
- Use the scripts in `scripts/` to download or generate datasets locally under `data/`.
- Example real-microscopy processing (outputs to `data/real_microscopy/`):

```
python3 scripts/process_microscopy_data.py \
  --wf-path /abs/path/to/wf.tif \
  --tp-path /abs/path/to/tp_reg.tif \
  --beads-dir /abs/path/to/beads \
  --output-dir /home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/real_microscopy \
  --create-splits
```

Quick start
-----------

1. Create and activate a Python virtual environment.
2. Install dependencies and the package in editable mode:

```
pip install -r requirements.txt
pip install -e .
```

3. Set up environment variables:

```bash
# Copy the example environment file
cp env.example .env

# Edit .env file and add your Weights & Biases API key
# Get your API key from: https://wandb.ai/settings
export WANDB_API_KEY=your_actual_api_key_here
```

4. Run tests:

```
python -m unittest -q
```

Data preparation
----------------

Use the provided CLIs to fetch a license-friendly ImageNet-like subset and aggregate images for training.

```
# 1) Download and prepare synthetic data (optional - for ImageNet-based experiments)
python3 scripts/preprocess_all.py --data-dir /home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data

# 3) (Optional) Synthesize WF/2P-like training pairs to disk
python3 scripts/synthesize_data.py \
  --source-dir /home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/train/classless \
  --output-dir /home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/synth \
  --image-size 256 --mode train
```

Notes:
The default dataset is the open Imagenette subset. Place any additional raw images under `data/raw/` if desired.

Project structure (Phase 1)
---------------------------

- pkl_dg/ (Python package)
- configs/
- scripts/
- tests/
- assets/
- docs/
- notebooks/

### Module overview and imports

Public API exported by subpackages:

- physics:
  - PSF
  - ForwardModel
  - PoissonNoise
  - GaussianBackground
- models:
  - DenoisingUNet
  - DDPMTrainer
- guidance:
  - GuidanceStrategy
  - PKLGuidance
  - L2Guidance
  - AnscombeGuidance
  - AdaptiveSchedule
 - evaluation:
   - Metrics
   - RobustnessTests

Example imports:

```
from pkl_dg.physics import PSF, ForwardModel, PoissonNoise, GaussianBackground
from pkl_dg.models import DenoisingUNet, DDPMTrainer
from pkl_dg.guidance import GuidanceStrategy, PKLGuidance, L2Guidance, AnscombeGuidance, AdaptiveSchedule
from pkl_dg.evaluation import Metrics, RobustnessTests
```



Guidance (Phase 5)
------------------

- Implemented guidance strategies under `pkl_dg/guidance`:
  - `GuidanceStrategy` (base interface)
  - `PKLGuidance`
  - `L2Guidance`
  - `AnscombeGuidance`
  - `AdaptiveSchedule`

Example usage:

```
import torch
from pkl_dg.guidance import PKLGuidance, L2Guidance, AnscombeGuidance, AdaptiveSchedule

# Minimal forward model with identity PSF for demonstration
class ForwardModel:
    def __init__(self, background: float = 0.0):
        self.background = background
    def apply_psf(self, x: torch.Tensor) -> torch.Tensor:
        return x
    def apply_psf_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y

# Dummy data
x0_hat = torch.rand(1, 1, 64, 64)
y = torch.rand(1, 1, 64, 64)
fm = ForwardModel(background=0.1)

# Choose a guidance strategy
guidance = PKLGuidance(epsilon=1e-6)

# Compute gradient and adaptive step size, then apply
grad = guidance.compute_gradient(x0_hat, y, fm, t=500)
sched = AdaptiveSchedule(lambda_base=0.1, T_threshold=800, T_total=1000)
lambda_t = sched.get_lambda_t(grad, t=500)
x0_hat = guidance.apply_guidance(x0_hat, grad, lambda_t)
```

Run guidance tests only:

```
pytest -q tests/test_guidance.py
```


Evaluation (Phase 7)
--------------------

- Metrics available under `pkl_dg/evaluation/metrics.py`:
  - `Metrics.psnr`, `Metrics.ssim`, `Metrics.frc`, `Metrics.sar`, `Metrics.hausdorff_distance`
- Robustness tests under `pkl_dg/evaluation/robustness.py`:
  - `RobustnessTests.psf_mismatch_test`, `RobustnessTests.alignment_error_test`

Example usage (metrics):

```
import numpy as np
from pkl_dg.evaluation import Metrics

rng = np.random.RandomState(0)
target = rng.rand(64, 64).astype(np.float32)
pred = target + 0.05 * rng.randn(64, 64).astype(np.float32)

psnr_val = Metrics.psnr(pred, target)          # in dB
ssim_val = Metrics.ssim(pred, target, data_range=1.0)
frc_res = Metrics.frc(pred, target, threshold=0.143)

# SAR and Hausdorff distance
artifact_mask = np.zeros_like(target, dtype=bool)
artifact_mask[:8, :8] = True
sar_db = Metrics.sar(pred, artifact_mask)

pred_mask = pred > 0.6
target_mask = target > 0.6
hd = Metrics.hausdorff_distance(pred_mask, target_mask)
```

Example usage (robustness tests):

```
import torch
from pkl_dg.evaluation import RobustnessTests
from pkl_dg.physics import PSF

# Assume `sampler` is an instance of pkl_dg.models.sampler.DDIMSampler
# and `y` is a measurement tensor with shape [B, C, H, W]
psf_true = PSF()
x_mismatch = RobustnessTests.psf_mismatch_test(sampler, y, psf_true, mismatch_factor=1.1)
x_shifted = RobustnessTests.alignment_error_test(sampler, y, shift_pixels=0.5)
```

Run evaluation tests only:

```
pytest -q tests/test_evaluation_*
```


Training (Phase 8)
------------------

- Training script: `scripts/train_diffusion.py`
- Behavior: lightweight routine (no Lightning Trainer) to keep tests fast and dependencies minimal. It builds datasets and `DDPMTrainer`, performs a quick sanity forward pass, and saves `checkpoints/final_model.pt`.
- Config: Hydra-based using `configs/`. Override via CLI as needed.

Run with W&B disabled:

```
python3 scripts/train_diffusion.py wandb.mode=disabled
```

Common overrides:

```
# fast local run
python3 scripts/train_diffusion.py \
  wandb.mode=disabled training.max_epochs=1 training.use_ema=false data.image_size=32
```

Expected dataset layout for `SynthesisDataset`:

```
data/
  train/  # .png/.jpg/.tif images
  val/    # .png/.jpg/.tif images
```

Hydra groups provided:

- `model/unet.yaml`
- `data/synthesis.yaml`
- `physics/microscopy.yaml`
- `training/ddpm.yaml`
- `guidance/pkl.yaml`

Note on test warnings: two warnings appear when unit tests invoke `DDPMTrainer.training_step` and `validation_step` directly (without a Lightning `Trainer`). PyTorch Lightning warns that `self.trainer` is not registered for `self.log()`. This is expected in tests and safe to ignore; it does not occur when using a real Trainer loop.


Evaluation (Includes Inference + Metrics)
------------------------------------------

- Script: `python -m pkl_dg.evaluation.evaluation`
- Configs: guidance settings in `configs/guidance/pkl.yaml`
- Inputs: directory of input images and ground truth images
- Outputs: quantitative metrics and comparison visualizations

The evaluation scripts perform inference with multiple guidance strategies (PKL, L2, Anscombe) and compute comprehensive metrics against ground truth.

Run evaluation with defaults:

```
python scripts/main.py evaluate \
  --checkpoint /abs/path/to/final_model.pt \
  --real-dir /abs/path/to/ground_truth \
  --fake-dir /abs/path/to/inputs
```

For baseline comparisons:

```
python scripts/main.py evaluate \
  --baseline all \
  --real-dir /abs/path/to/ground_truth \
  --fake-dir /abs/path/to/inputs
```

