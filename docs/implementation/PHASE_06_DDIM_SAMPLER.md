# Phase 6: DDIM Sampler with Guidance (Week 4-5)

### Step 6.1: Implement DDIM Sampler
```python
# pkl_dg/models/sampler.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tqdm import tqdm

class DDIMSampler:
    """DDIM sampler with guidance injection."""
    
    def __init__(
        self,
        model: nn.Module,              # DDPMTrainer (LightningModule) exposing buffers
        forward_model: 'ForwardModel',
        guidance_strategy: 'GuidanceStrategy',
        schedule: 'AdaptiveSchedule',
        transform: 'IntensityToModel',
        num_timesteps: int = 1000,
        ddim_steps: int = 100,
        eta: float = 0.0
    ):
        self.model = model
        self.forward_model = forward_model
        self.guidance = guidance_strategy
        self.schedule = schedule
        self.transform = transform
        self.num_timesteps = num_timesteps
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.ddim_timesteps = self._setup_ddim_timesteps()
    
    def _setup_ddim_timesteps(self):
        c = self.num_timesteps // self.ddim_steps
        return torch.tensor(list(range(0, self.num_timesteps, c))[::-1])
    
    @torch.no_grad()
    def sample(self, y: torch.Tensor, shape: tuple, device: str = 'cuda', verbose: bool = True) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        y = y.to(device)
        iterator = tqdm(self.ddim_timesteps, desc="DDIM Sampling") if verbose else self.ddim_timesteps
        for i, t in enumerate(iterator):
            t_cur = t
            t_next = self.ddim_timesteps[i + 1] if i < len(self.ddim_timesteps) - 1 else 0
            x0_hat = self._predict_x0(x_t, t_cur)
            if t_cur > 0:
                x0_hat_corrected = self._apply_guidance(x0_hat, y, t_cur)
            else:
                x0_hat_corrected = x0_hat
            x_t = self._ddim_step(x_t, x0_hat_corrected, t_cur, t_next)
        x0_final = self._predict_x0(x_t, torch.tensor(0, device=device))
        x0_intensity = self.transform.inverse(x0_final)
        return x0_intensity
    
    def _predict_x0(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x_t.device)
        if t.dim() == 0:
            t = t.repeat(x_t.shape[0])
        alpha_t = self.model.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        # Use EMA model if available
        net = self.model.ema_model if getattr(self.model, 'use_ema', False) else self.model.model
        noise_pred = net(x_t, t)
        x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        return x0_hat
    
    def _apply_guidance(self, x0_hat: torch.Tensor, y: torch.Tensor, t: int) -> torch.Tensor:
        x0_intensity = self.transform.inverse(x0_hat)
        x0_intensity = torch.clamp(x0_intensity, min=0)
        gradient = self.guidance.compute_gradient(x0_intensity, y, self.forward_model, t)
        lambda_t = self.schedule.get_lambda_t(gradient, t)
        x0_corrected = self.guidance.apply_guidance(x0_intensity, gradient, lambda_t)
        x0_corrected = torch.clamp(x0_corrected, min=0)
        x0_corrected_model = self.transform(x0_corrected)
        return x0_corrected_model
    
    def _ddim_step(self, x_t: torch.Tensor, x0_hat: torch.Tensor, t_cur: int, t_next: int) -> torch.Tensor:
        alpha_cur = self.model.alphas_cumprod[t_cur]
        alpha_next = self.model.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=x_t.device)
        sigma_t = self.eta * torch.sqrt((1 - alpha_next) / (1 - alpha_cur)) * torch.sqrt(1 - alpha_cur / alpha_next)
        sqrt_one_minus_alpha_cur = torch.sqrt(1 - alpha_cur)
        pred_noise = (x_t - torch.sqrt(alpha_cur) * x0_hat) / sqrt_one_minus_alpha_cur
        sqrt_alpha_next = torch.sqrt(alpha_next)
        dir_x_t = torch.sqrt(1 - alpha_next - sigma_t**2) * pred_noise
        if t_next > 0:
            noise = torch.randn_like(x_t)
            x_next = sqrt_alpha_next * x0_hat + dir_x_t + sigma_t * noise
        else:
            x_next = x0_hat
        return x_next
```
