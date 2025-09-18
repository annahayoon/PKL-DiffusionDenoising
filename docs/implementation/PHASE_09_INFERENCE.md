# Phase 9: Inference Script (Week 6)

### Step 9.1: Create Inference Script
```python
# scripts/inference.py
import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import tifffile
import numpy as np
from tqdm import tqdm

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.guidance.pkl_optical_restoration import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.data.transforms import IntensityToModel

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def inference(cfg: DictConfig):
    """Run guided diffusion inference."""
    
    # Setup device
    device = cfg.experiment.device
    
    # Load model
    print("Loading model...")
    unet = DenoisingUNet(cfg.model)
    ddpm = DDPMTrainer(unet, cfg.training)
    
    checkpoint_path = Path(cfg.inference.checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(state_dict)
    ddpm.eval()
    ddpm.to(device)
    
    # Use EMA model if available
    if ddpm.use_ema:
        model = ddpm.ema_model
    else:
        model = ddpm.model
    
    # Setup forward model
    psf = PSF(cfg.physics.psf_path)
    forward_model = ForwardModel(
        psf=psf.to_torch(device=device),
        background=cfg.physics.background,
        device=device
    )
    
    # Setup guidance
    guidance_type = cfg.guidance.type
    if guidance_type == 'pkl':
        guidance = PKLGuidance(epsilon=cfg.guidance.epsilon)
    elif guidance_type == 'l2':
        guidance = L2Guidance()
    elif guidance_type == 'anscombe':
        guidance = AnscombeGuidance(epsilon=cfg.guidance.epsilon)
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")
    
    # Setup schedule
    schedule = AdaptiveSchedule(
        lambda_base=cfg.guidance.lambda_base,
        T_threshold=cfg.guidance.schedule.T_threshold,
        epsilon_lambda=cfg.guidance.schedule.epsilon_lambda,
        T_total=cfg.training.num_timesteps
    )
    
    # Setup transform
    transform = IntensityToModel(
        min_intensity=cfg.data.min_intensity,
        max_intensity=cfg.data.max_intensity
    )
    
    # Create sampler
    sampler = DDIMSampler(
        model=ddpm,
        forward_model=forward_model,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=cfg.training.num_timesteps,
        ddim_steps=cfg.inference.ddim_steps,
        eta=cfg.inference.eta
    )
    
    # Process input images
    input_dir = Path(cfg.inference.input_dir)
    output_dir = Path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all input images
    image_paths = list(input_dir.glob("*.tif"))
    image_paths += list(input_dir.glob("*.tiff"))
    
    print(f"Processing {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths):
        # Load measurement
        y = tifffile.imread(img_path)
        y = torch.from_numpy(y).float().to(device)
        
        # Ensure right shape
        if y.ndim == 2:
            y = y.unsqueeze(0).unsqueeze(0)
        
        # Run reconstruction
        shape = y.shape
        reconstruction = sampler.sample(y, shape, device=device, verbose=False)
        
        # Save result
        output_path = output_dir / f"{img_path.stem}_reconstructed.tif"
        reconstruction_np = reconstruction.squeeze().cpu().numpy()
        tifffile.imwrite(output_path, reconstruction_np.astype(np.float32))
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    inference()
```
