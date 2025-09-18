import os
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import tifffile
from tqdm import tqdm
from PIL import Image
import wandb

from pkl_dg.models.unet import UNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.physics.psf import build_psf_bank
 
from pkl_dg.guidance.pkl_optical_restoration import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.data import IntensityToModel, AnscombeToModel, GeneralizedAnscombeToModel
from pkl_dg.baselines import richardson_lucy_restore

# Import new metrics system
from .metrics_pkg import compute_metrics, compute_downstream_metrics, list_metrics
from .metrics_pkg.robustness import test_alignment_error_with_sampler, add_out_of_focus_artifact
from .schemas import EvaluationConfig, validate_config

try:
    from pkl_dg.baselines import RCANWrapper  # optional
    HAS_RCAN = True
except Exception:
    HAS_RCAN = False


def _load_model_and_sampler(cfg: DictConfig, guidance_type: str):
    device = str(cfg.experiment.device)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", False))
    unet = UNet(model_cfg)
    
    # Create forward model for self-supervised trainer
    psf_config = getattr(cfg, "psf", {})
    if psf_config.get("type") == "gaussian":
        psf = PSF(
            sigma_x=float(psf_config.get("sigma_x", 2.0)),
            sigma_y=float(psf_config.get("sigma_y", 2.0)),
            size=int(psf_config.get("size", 21))
        )
    else:
        psf = PSF()
    
    forward_model = ForwardModel(
        psf=psf.to_torch(device=device),
        background=float(psf_config.get("background", 0.0)),
        device=device,
        common_sizes=[(int(getattr(cfg.data, "image_size", 256)), int(getattr(cfg.data, "image_size", 256)))]
    )
    
    # Create DDPM trainer
    ddpm = DDPMTrainer(
        model=unet,
        config=OmegaConf.to_container(cfg.training, resolve=True),
        forward_model=forward_model,
    )
    checkpoint_path = Path(str(cfg.inference.checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Load state dict (includes EMA weights if present)
    ddpm.load_state_dict(state_dict, strict=False)
    ddpm.eval().to(device)
    # Forward model per paper (PSF + background)
    forward_model = None
    try:
        phys_cfg = cfg.physics
        use_psf = bool(getattr(phys_cfg, "use_psf", False))
        background = float(getattr(phys_cfg, "background", 0.0))
        if use_psf:
            psf_path = getattr(phys_cfg, "psf_path", None)
            use_bead = bool(getattr(phys_cfg, "use_bead_psf", False))
            if use_bead:
                beads_dir = str(getattr(phys_cfg, "beads_dir", ""))
                bank = build_psf_bank(beads_dir)
                mode = getattr(phys_cfg, "bead_mode", None)
                if mode is None:
                    # Heuristic: prefer with_AO if available
                    psf_t = bank.get("with_AO", bank.get("no_AO"))
                else:
                    psf_t = bank.get(str(mode), next(iter(bank.values())))
                psf = psf_t.to(device=device, dtype=torch.float32)
                if psf.ndim == 2:
                    psf = psf.unsqueeze(0).unsqueeze(0)
            elif psf_path is not None:
                psf = PSF(psf_path=str(psf_path)).to_torch(device=device)
            else:
                # Default Gaussian PSF if none provided
                psf = PSF().to_torch(device=device)
            read_noise_sigma = float(getattr(phys_cfg, "read_noise_sigma", 0.0))
            
            # Get pixel size configuration for PSF scaling
            target_pixel_size_xy_nm = getattr(phys_cfg, "target_pixel_size_xy_nm", None)
            if target_pixel_size_xy_nm is not None:
                target_pixel_size_xy_nm = float(target_pixel_size_xy_nm)
            
            forward_model = ForwardModel(
                psf=psf, 
                background=background, 
                device=device, 
                read_noise_sigma=read_noise_sigma,
                target_pixel_size_xy_nm=target_pixel_size_xy_nm
            )
    except Exception:
        forward_model = None
    if guidance_type == "pkl":
        guidance = PKLGuidance(epsilon=float(getattr(cfg.guidance, "epsilon", 1e-6)))
    elif guidance_type == "l2":
        guidance = L2Guidance()
    elif guidance_type == "anscombe":
        guidance = AnscombeGuidance(epsilon=float(getattr(cfg.guidance, "epsilon", 1e-6)))
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")
    schedule_cfg = getattr(cfg.guidance, "schedule", {})
    schedule = AdaptiveSchedule(
        lambda_base=float(getattr(cfg.guidance, "lambda_base", 0.1)),
        T_threshold=int(getattr(schedule_cfg, "T_threshold", 800)),
        epsilon_lambda=float(getattr(schedule_cfg, "epsilon_lambda", 1e-3)),
        T_total=int(cfg.training.num_timesteps),
    )
    noise_model = str(getattr(cfg.data, "noise_model", "gaussian")).lower()
    if noise_model == "poisson":
        transform = AnscombeToModel(maxIntensity=float(cfg.data.max_intensity))
    elif noise_model == "poisson_gaussian":
        gat_cfg = getattr(cfg.data, "gat", {})
        transform = GeneralizedAnscombeToModel(
            maxIntensity=float(cfg.data.max_intensity),
            alpha=float(getattr(gat_cfg, "alpha", 1.0)),
            mu=float(getattr(gat_cfg, "mu", 0.0)),
            sigma=float(getattr(gat_cfg, "sigma", 0.0)),
        )
    else:
        transform = IntensityToModel(min_intensity=float(cfg.data.min_intensity), max_intensity=float(cfg.data.max_intensity))
    sampler = DDIMSampler(
        model=ddpm,
        forward_model=forward_model,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=int(cfg.training.num_timesteps),
        ddim_steps=int(cfg.inference.ddim_steps),
        eta=float(cfg.inference.eta),
        use_autocast=bool(getattr(cfg.inference, "use_autocast", True)),
    )
    return sampler


def _compute_metrics(pred: np.ndarray, target: np.ndarray, input_img: Optional[np.ndarray] = None, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute metrics using the new registry system."""
    pr = pred.astype(np.float32)
    gt = target.astype(np.float32)
    inp = input_img.astype(np.float32) if input_img is not None else None
    
    if metric_names is None:
        # Use default metrics for backward compatibility
        metric_names = ['psnr', 'ssim', 'frc']
        if inp is not None:
            metric_names.append('sar')
    
    data_range = float(gt.max() - gt.min()) if gt.size > 0 else 1.0
    
    return compute_metrics(
        pred=pr,
        target=gt,
        input_img=inp,
        metric_names=metric_names,
        data_range=data_range,
        threshold=0.143  # FRC threshold
    )


def _compute_downstream_metrics(
    pred: np.ndarray, gt_masks: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for downstream tasks like segmentation."""
    return compute_downstream_metrics(pred, gt_masks)


def _load_pairs(input_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
        paths.extend(list(input_dir.glob(ext)))
    return sorted(paths)


def _read_tif(path: Path) -> np.ndarray:
    if path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        return np.array(Image.open(path)).astype(np.float32)
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


def _save_visuals(out_dir: Path, name: str, wf: np.ndarray, pred_map: Dict[str, np.ndarray], gt: Optional[np.ndarray] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Normalize to 0-255 for visualization per-image
    def norm(img: np.ndarray) -> np.ndarray:
        a = img.astype(np.float32)
        lo, hi = np.quantile(a, 0.01), np.quantile(a, 0.99)
        if hi <= lo:
            hi = lo + 1.0
        a = (a - lo) / (hi - lo)
        a = np.clip(a, 0, 1)
        return (a * 255.0).astype(np.uint8)

    grid_parts: List[np.ndarray] = [norm(wf)]
    for k in sorted(pred_map.keys()):
        grid_parts.append(norm(pred_map[k]))
    if gt is not None:
        grid_parts.append(norm(gt))

    # Concatenate horizontally
    grid = np.concatenate(grid_parts, axis=1)
    Image.fromarray(grid).save(out_dir / f"{name}_comparison.png")


def evaluate(cfg: DictConfig) -> Dict[str, Dict[str, float]]:
    # Validate configuration
    try:
        validate_config(cfg)
    except Exception as e:
        print(f"⚠️ Configuration validation failed: {e}")
        print("Continuing with potentially invalid configuration...")
    
    # Initialize wandb if enabled
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.experiment.name}_evaluation",
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags + ["evaluation"],
            notes=cfg.wandb.notes,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type
        )
        print("✅ Initialized Weights & Biases logging")
    
    device = str(cfg.experiment.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_dir = Path(str(cfg.inference.input_dir))
    gt_dir = Path(str(cfg.inference.gt_dir))
    mask_dir = Path(str(getattr(cfg.inference, "mask_dir", "")))
    image_paths = _load_pairs(input_dir)
    
    print(f"Device: {device}")
    print(f"Found {len(image_paths)} images to evaluate")
    
    # Determine which metrics to compute
    all_metrics = []
    if hasattr(cfg, 'metrics'):
        all_metrics.extend(getattr(cfg.metrics, 'image_quality', []))
        all_metrics.extend(getattr(cfg.metrics, 'perceptual', []))
        all_metrics.extend(getattr(cfg.metrics, 'robustness', []))
    
    if not all_metrics:
        # Fallback to default metrics
        all_metrics = ['psnr', 'ssim', 'frc', 'sar', 'sharpness', 'contrast']
    
    print(f"Computing metrics: {all_metrics}")

    # Prepare samplers per guidance
    conditioning_type = str(getattr(cfg.training, "conditioning_type", "wf")).lower()
    samplers = {
        "l2": _load_model_and_sampler(cfg, "l2"),
        "anscombe": _load_model_and_sampler(cfg, "anscombe"),
        "pkl": _load_model_and_sampler(cfg, "pkl"),
    }

    # Optional RCAN
    rcan = None
    baselines_cfg = None
    try:
        baselines_cfg = cfg.baselines  # may not exist
    except Exception:
        baselines_cfg = None
    if HAS_RCAN and baselines_cfg is not None and getattr(baselines_cfg, "rcan_checkpoint", None) is not None:
        try:
            rcan = RCANWrapper(checkpoint_path=str(baselines_cfg.rcan_checkpoint), device=device)
        except Exception:
            rcan = None

    # Results accumulators
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    def _acc(name: str, metrics: Dict[str, float]):
        if not metrics:
            return
        if name not in sums:
            sums[name] = {k: 0.0 for k in metrics}
            counts[name] = 0
        for k, v in metrics.items():
            if np.isfinite(v):
                sums[name][k] += float(v)
        counts[name] += 1

    for img_path in tqdm(image_paths, desc="Evaluate"):
        y = _read_tif(img_path)
        x_gt = _read_tif(gt_dir / img_path.name)
        
        # Load masks for downstream tasks if available
        gt_masks = None
        if mask_dir.is_dir():
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                gt_masks = _read_tif(mask_path)

        # WF input baseline
        wf_metrics = _compute_metrics(y, x_gt, input_img=y, metric_names=all_metrics)
        _acc("wf", wf_metrics)
        if gt_masks is not None:
            downstream_metrics = _compute_downstream_metrics(y, gt_masks)
            _acc("wf", downstream_metrics)
            
        # RL baseline removed (no PSF usage)
            
        # Diffusion baselines
        for name, sampler in samplers.items():
            ten_y = torch.from_numpy(y).float().to(device)
            if ten_y.ndim == 2:
                ten_y = ten_y.unsqueeze(0).unsqueeze(0)
            # Build conditioner in model domain if model is conditioned (in_channels=2)
            try:
                cond = sampler.transform(ten_y) if conditioning_type == "wf" else None
            except Exception:
                cond = None
            pred = sampler.sample(ten_y, tuple(ten_y.shape), device=device, verbose=False, conditioner=cond)
            out = pred.squeeze().detach().cpu().numpy().astype(np.float32)
            method_metrics = _compute_metrics(out, x_gt, input_img=y, metric_names=all_metrics)
            _acc(name, method_metrics)
            if gt_masks is not None:
                downstream_metrics = _compute_downstream_metrics(out, gt_masks)
                _acc(name, downstream_metrics)
        
        # Save side-by-side visuals (WF | l2 | anscombe | pkl | GT)
        preds = {}
        for k, sampler in samplers.items():
            ten_y_vis = torch.from_numpy(y).float().to(device).unsqueeze(0).unsqueeze(0)
            try:
                cond_vis = sampler.transform(ten_y_vis) if conditioning_type == "wf" else None
            except Exception:
                cond_vis = None
            out_vis = sampler.sample(ten_y_vis, (1, 1) + y.shape, device=device, verbose=False, conditioner=cond_vis)
            preds[k] = out_vis.squeeze().detach().cpu().numpy().astype(np.float32)
        _save_visuals(Path(str(cfg.inference.output_dir)) / "comparisons", img_path.stem, y, preds, x_gt)

        # RCAN if available
        if rcan is not None:
            try:
                rcan_out = rcan.infer(y)
                rcan_metrics = _compute_metrics(rcan_out, x_gt, input_img=y, metric_names=all_metrics)
                _acc("rcan", rcan_metrics)
                if gt_masks is not None:
                    downstream_metrics = _compute_downstream_metrics(rcan_out, gt_masks)
                    _acc("rcan", downstream_metrics)
            except Exception:
                pass

        # --- Adversarial Evaluations --- (PSF-dependent tests removed)

        # Robustness: Alignment Error
        try:
            x_shifted = test_alignment_error_with_sampler(samplers["pkl"], ten_y, shift_pixels=0.5)
            if isinstance(x_shifted, torch.Tensor):
                out = x_shifted.squeeze().detach().cpu().numpy().astype(np.float32)
            else:
                out = x_shifted.astype(np.float32)
            alignment_metrics = _compute_metrics(out, x_gt, input_img=y, metric_names=['psnr', 'ssim'])
            _acc("pkl_alignment_error", alignment_metrics)
        except Exception:
            pass

        # Hallucination: Commission Error (SAR)
        try:
            art_img, art_mask = add_out_of_focus_artifact(y, center=(y.shape[0]//2, y.shape[1]//2))
            ten_art = torch.from_numpy(art_img).float().to(device).unsqueeze(0).unsqueeze(0)
            out = samplers["pkl"].sample(ten_art, ten_art.shape, device=device, verbose=False)
            out_np = out.squeeze().detach().cpu().numpy().astype(np.float32)
            
            # Use the new metrics system for commission SAR
            commission_metrics = compute_metrics(
                pred=out_np,
                target=None,
                input_img=None,
                metric_names=['commission_sar'],
                artifact_mask=art_mask
            )
            _acc("pkl_commission_sar", commission_metrics)
        except Exception:
            pass
            
        # Hallucination: Omission Error removed (requires forward model)


    # Aggregate means
    results: Dict[str, Dict[str, float]] = {}
    for name, agg in sums.items():
        denom = max(counts.get(name, 1), 1)
        results[name] = {k: v / denom for k, v in agg.items()}
    
    # Log results to wandb if enabled
    if cfg.wandb.mode != "disabled":
        try:
            # Flatten results for wandb logging
            wandb_metrics = {}
            for method_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    wandb_metrics[f"{method_name}/{metric_name}"] = value
            
            wandb.log(wandb_metrics)
            
            # Create summary table
            import pandas as pd
            summary_data = []
            for method_name, metrics in results.items():
                row = {"method": method_name}
                row.update(metrics)
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            wandb.log({"results_table": wandb.Table(dataframe=df)})
            
            print("✅ Logged results to Weights & Biases")
            
        except Exception as e:
            print(f"⚠️ Failed to log to wandb: {e}")
    
    return results


# Register the structured config
cs = ConfigStore.instance()
cs.store(name="evaluation_config", node=EvaluationConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    try:
        print("🚀 Starting Evaluation")
        print("=" * 50)
        
        res = evaluate(cfg)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 60)
        
        for name, metrics in res.items():
            print(f"\n{name.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric:20}: {value:8.4f}")
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {cfg.inference.output_dir}")
        
        # Print available metrics info
        print(f"\nAvailable metrics: {list_metrics()}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise
    finally:
        # Clean up wandb
        if cfg.wandb.mode != "disabled":
            try:
                wandb.finish()
                print("✅ Weights & Biases session closed")
            except:
                pass


if __name__ == "__main__":
    main()


