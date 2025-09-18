## Updated Design: Microscopy Diffusion Training/Evaluation (`scripts/run_microscopy.py` + `configs/multi_gpu_config.yaml`)

### 0) Primary mapping and PSF usage
- **Primary mapping**: Default is an unconditional diffusion prior over 2P; WF enters primarily via the physics model (guidance/consistency). The PSF is external to the UNet. Optional explicit WF conditioning is supported by setting `training.use_conditioning=true` (then `UNet` concatenates a conditioner channel), but is disabled by default (`training.use_conditioning=false`).
- **Where PSF is used**:
  - Training consistency (self-supervised, optional/light): simulate WF by convolving the (predicted) 2P image with the PSF via `ForwardModel`; define a small loss in the WF domain. Enabled via `training.use_forward_consistency` with a small `forward_consistency_weight` and warmup.
  - Inference guidance: L2/KL/PKL guidance compares the forward projection of the current 2P estimate against the observed WF to steer sampling.
  - Classical baseline: Richardson–Lucy deconvolution uses the PSF as a non-learning comparator.

### 0.1) Novelty: Poisson‑guided KL guidance (PKL)
- **What**: A KL-divergence guidance term derived from the Poisson photon-counting likelihood couples the `ForwardModel` with the sampler.
- **Why**: Photon-limited microscopy is better modeled by Poisson statistics than Gaussian. PKL leverages this to provide statistically grounded guidance that outperforms L2 under low SNR.
- **How**: During sampling, compare the Poisson likelihood of the forward-projected prediction with the observed WF and backpropagate its KL gradient to guide updates. Our implementation includes an optional Gaussian read-noise term in the denominator (σ²) for robustness in mixed noise.

### 1) Project Goal
- **Objective**: Train and evaluate denoising diffusion models that convert wide-field (WF) microscopy images into two-photon (2P)-like reconstructions using self-supervised learning augmented with a physics-based forward model.
- **Key ideas**:
  - Physics-consistent simulation: simulate WF by convolving 2P with a measured/parametric PSF; the model ultimately predicts 2P from WF.
  - Multi-GPU, mixed-precision training for scale (A40 GPUs/cluster-ready).
  - Evaluation with standard imaging metrics and a classical RL baseline.

### 2) Architecture
#### 2.1 Data flow overview
- Training mode
  - Inputs: `cfg` (Hydra YAML), WF/2P datasets at `paths.data`, optional measured PSF (beads) or Gaussian PSF.
  - Process: datasets → dataloaders → `UNet` prior (unconditional 2P) → `DDPMTrainer` losses (DDPM + optional forward-consistency via `ForwardModel`) → checkpoints/logs.
  - Outputs: model checkpoints, logs (`logs/`, `lightning_logs/`), optional periodic full-frame reconstructions, W&B runs.
- Evaluation mode
  - Inputs: `checkpoint`, input directory (`wf`), ground-truth directory (`2p`), guidance type(s), PSF (optional for baselines).
  - Process: restore model → build sampler(s) with chosen guidance (uses `ForwardModel`) → sample → compute metrics (+ baselines).
  - Outputs: aggregated metrics to stdout and `evaluation_results.json`.

#### 2.2 Model
- **Network**: `pkl_dg.models.unet.UNet` instantiated from `cfg.model`.
  - `in_channels` may be increased to 2 if conditioning is enabled (`x_t + WF conditioner`), controlled by `training.use_conditioning` (default false).
  - `multi_gpu_config.yaml` specifies a moderately deep U-Net with attention in mid/high-resolution blocks for 128×128 training.
- **Trainer**: `pkl_dg.models.diffusion.DDPMTrainer` encapsulates the diffusion objective(s), optional EMA, and training/eval steps.
  - Receives `forward_model` and `transform` for self-supervised training in microscopy.

#### 2.3 Loss functions
- Core DDPM objective (simple noise-prediction loss) governed by `cfg.training`.
  - `ddpm_loss_weight` defaults to 1.0 if not provided.
- Optional forward-consistency term (self-supervised, light): simulate WF via `ForwardModel(\hat x_0)` and compare to observed WF in the measurement domain.
  - Controlled by: `training.use_forward_consistency`, `training.forward_consistency_weight` (small, e.g., 0.01), `training.forward_consistency_type` (`l2|kl|pkl`), and `training.forward_consistency_warmup_steps`.
- Learned variance is supported if enabled in `cfg.model.learned_variance` and corresponding trainer logic.
- Poisson-aware ELBO: implemented in `DDPMTrainer` and, when enabled (`training.use_elbo_loss=true`), is combined with the simple objective (`combine_elbo_with_simple=true`). The t=0 observation term uses a Poisson NLL with small epsilon for stability.
- Perceptual loss is off by default for pure self-supervision.

#### 2.4 Physics-based forward model and guidance
- **PSF/Forward model**
  - Sources: measured PSF from beads (preferred) or Gaussian fallback from `cfg.psf`.
  - Wrapped in `pkl_dg.physics.PSF`, consumed by `pkl_dg.physics.ForwardModel` with optional pixel-size scaling when PSF and image pixel sizes differ.
  - Background `B` comes from `cfg.physics.background`; optional read-noise `σ_read` is wired via `cfg.physics.read_noise_sigma` and used by PKL guidance.
  - Training: optional small forward-consistency by convolving (predicted) 2P with PSF to form a loss in the WF domain.
- **Guidance at inference**
  - Supported: **Poisson‑guided KL (PKL)**, plain **KL**, and **L2** (with optional Anscombe variance stabilization).
  - `load_model_and_sampler` builds a sampler using a stable DDIM scheduler (DPMSolver++ disabled by default). Guidance compares forward projections to observed WF each step in the x₀ domain.
  - Configure via YAML (`guidance.type: pkl|kl|l2|anscombe`) or override at CLI with `--guidance-type {pkl,kl,l2,anscombe}`; if unset, evaluation can run multiple guidance types.
  - Adaptive schedule: `guidance.schedule_type=adaptive` with `lambda_base`, `schedule.T_threshold`, and `schedule.epsilon_lambda` normalizes by the gradient norm and ramps guidance late for stability. These can be swept in ablations via `--sweep-guidance-lambda`, `--sweep-guidance-Tthr`, and `--sweep-guidance-epslambda`.

Note on defaults
- The default configs train an unconditional 2P prior and enable a small forward-consistency regularizer (weight ≈ 0.01 with warmup). WF is used strongly at inference via physics-guided sampling; default guidance is `pkl` from YAML unless overridden at the CLI.

### 3) Features
- **Unified CLI** (`scripts/run_microscopy.py`)
  - Modes: `train`, `eval`, `train_eval`, `ablate`.
  - Hydra-based config loading with CLI overrides (epochs/steps, batch, LR, device, seed, paths, W&B mode).
- **Data handling**
  - Unpaired self-supervised WF/2P folders: `${paths.data}/{train,val}/{wf,2p}`.
  - 16-bit safe transforms: `Microscopy16BitToModel` or `IntensityToModel` with inverse mapping used to de/normalize between model [-1,1] and sensor intensities (clamped to ≥0 for physics ops).
  - Optional adaptive normalization (percentile-based) with its own loaders.
- **Training**
  - PyTorch Lightning trainer with DDP/multi-GPU (`hardware.devices`, `hardware.strategy`, `hardware.accelerator`).
  - Mixed precision (fp16/bf16) via `training.precision` / `experiment.mixed_precision`.
  - Step-based validation/checkpointing (e.g., validate every N steps, save every 10k steps).
  - Early stopping on validation loss, gradient clipping, EMA (if available), cuDNN benchmarking, TF32 on A40.
- **Logging/Outputs**
  - File logger to `cfg.paths.logs`.
  - Lightning CSV logs under `lightning_logs/`.
  - W&B integration (offline/online/disabled).
  - Optional periodic full-frame reconstructions to `cfg.paths.outputs/reconstructions`.
- **Evaluation & Ablations**
  - Runs the selected guidance sampler (`guidance.type` or `--guidance-type`) and aggregates PSNR/SSIM/FRC. If no type is specified, the script can evaluate multiple guidance types.
  - Optional RL baseline when PSF is available.
  - Robustness tests: optional misalignment and PSF broadening tests via `--include-robustness-tests`.
  - Downstream metrics: optional Cellpose F1 (IoU≥0.5 fallback) and Hausdorff via `--include-cellpose --gt-masks-dir <dir>`.
  - Saves `evaluation_results.{json,csv}` to `--output-dir`.
  - Ablation sweeps via `--mode ablate` with CLI axes over guidance, PSF source, conditioning, adaptive normalization, `num_timesteps`, learned variance, EMA, `cycle_loss_weight`, and guidance schedule params (`lambda_base`, `T_threshold`, `epsilon_lambda`). Each combo saves results under `outputs/ablations/<config-stamped-run>/evaluation_results.{json,csv}` and writes aggregated `ablations_<config_name>_<timestamp>.{csv,json}`.

### 4) Baselines
- **Wide-field passthrough**: raw WF vs GT (sanity baseline).
- **Richardson–Lucy deconvolution** (`pkl_dg.baseline.richardson_lucy_restore`) using PSF; metrics via the same pipeline.

### Input → Output mapping
- `setup_experiment(args, cfg=None)`
  - **Input**: CLI `args` (including `--config`), optional `cfg`.
  - **Process**: Hydra load/compose → apply CLI overrides (epochs/steps/batch/LR/device/seed, adaptive normalization, paths, W&B mode) → optional summary.
  - **Output**: resolved `DictConfig cfg`.

- `run_training(cfg, args)`
  - **Input**: `cfg`, `args`.
  - **Process**: seeds/CUDA perf; build `ForwardModel` (PSF) and transforms; datasets/dataloaders; instantiate `UNet`+`DDPMTrainer`; optimizer (AdamW), AMP; Lightning training with DDP; step-based val/ckpt/early-stop; optional reconstructions.
  - **Output**: trained `DDPMTrainer`; artifacts: checkpoints, logs, CSV metrics, optional reconstructions.

- `load_model_and_sampler(cfg, checkpoint_path, guidance_type, device)`
  - **Input**: `cfg`, checkpoint, `guidance_type` (`pkl|kl|l2|anscombe`), device.
  - **Process**: restore UNet; build `ForwardModel` (PSF); create guidance; create `DDIMSampler` with `num_timesteps`.
  - **Output**: sampler ready for WF-conditioned denoising.

- `run_evaluation(cfg, args)`
  - **Input**: `cfg`, `--input-dir`, `--gt-dir`, `--checkpoint`, `--include-baselines`, optional `--guidance-type`.
  - **Process**: read/normalize images; build sampler(s) for requested guidance; sample with WF conditioning; compute metrics; optional RL baseline; aggregate.
  - **Output**: averaged metrics printed and saved to `--output-dir/evaluation_results.{json,csv}`.

- `run_ablation(cfg, args)`
  - **Input**: `cfg`, eval paths, and sweep args: `--sweep-guidance`, `--sweep-psf-source`, `--sweep-conditioning`, `--sweep-adaptive-normalization`, `--sweep-num-timesteps`, `--sweep-learned-variance`, `--sweep-ema`, `--sweep-cycle-weight`.
  - **Process**: generate Cartesian product of sweeps, apply to config, run evaluation per combo, save per-run results and aggregate across runs.
  - **Output**: per-run `evaluation_results.{json,csv}` under `outputs/ablations/<run>/`; aggregated `ablations_<config_name>_<timestamp>.{csv,json}`.

### How `multi_gpu_config.yaml` pairs with the script
- Provides defaults for `experiment.*`, `paths.*`, `wandb.*`, `model.*`, `training.*`, `psf.*`/`physics.*`, `guidance.*`, and `hardware.*`.
- Advanced fields (progressive/multi-resolution, advanced schedulers, compile/fused optimizers, augmentation, logging.* and optimization.*) require additional wiring to take effect.

### Quick-start examples
- Train (multi-GPU cluster, using YAML paths)
```bash
python scripts/run_microscopy.py --mode train --config configs/multi_gpu_config.yaml --wandb-mode offline
```
- Evaluate (PKL guidance)
```bash
python scripts/run_microscopy.py --mode eval \
  --checkpoint /global/scratch/users/anna_yoon/checkpoints/best_model.pt \
  --input-dir /global/scratch/users/anna_yoon/data/real_microscopy/val/wf \
  --gt-dir /global/scratch/users/anna_yoon/data/real_microscopy/val/2p \
  --include-baselines --output-dir /global/scratch/users/anna_yoon/outputs/eval
```
- Evaluate with L2 or KL guidance
```bash
python scripts/run_microscopy.py --mode eval \
  --config configs/multi_gpu_config.yaml \
  --guidance-type l2

python scripts/run_microscopy.py --mode eval \
  --config configs/multi_gpu_config.yaml \
  --guidance-type kl
```

- Run ablations (example)
```bash
python scripts/run_microscopy.py --mode ablate \
  --config configs/multi_gpu_config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --input-dir data/test/wf --gt-dir data/test/2p \
  --output-dir outputs \
  --sweep-guidance pkl,kl,l2 \
  --sweep-psf-source beads,gaussian \
  --sweep-conditioning true,false \
  --sweep-adaptive-normalization false \
  --sweep-num-timesteps 250,500,1000 \
  --sweep-learned-variance true,false \
  --sweep-ema true,false \
  --sweep-cycle-weight 0.05,0.1,0.2
```
