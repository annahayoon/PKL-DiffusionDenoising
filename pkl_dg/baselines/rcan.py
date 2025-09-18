from typing import Optional

import numpy as np


class RCANWrapper:
    """Thin wrapper to run RCAN if available.

    Attempts to import `rcan` or `basicsr` style implementations. This is a best-effort
    convenience for evaluation; if dependencies are missing, it raises ImportError with
    a clear message.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: Optional[str]):
        try:
            import torch
            # Try a common RCAN implementation
            try:
                from rcan.model import RCAN as RCANNet  # type: ignore
            except Exception:
                # Fallback to a BasicSR-like RCAN if available
                from basicsr.archs.rcan_arch import RCAN as RCANNet  # type: ignore

            # Minimal RCAN config; real configs should mirror training
            model = RCANNet()
            if checkpoint_path is not None:
                state = torch.load(checkpoint_path, map_location=self.device)
                state_dict = state.get("state_dict", state)
                model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(self.device)
            self.model = model
        except Exception as e:
            raise ImportError(
                "RCAN dependencies not available. Install an RCAN implementation (e.g., rcan or basicsr) and provide a compatible checkpoint."
            ) from e

    @staticmethod
    def _to_tensor(img: np.ndarray, device: str):
        import torch

        ten = torch.from_numpy(img.astype(np.float32))
        if ten.ndim == 2:
            ten = ten.unsqueeze(0).unsqueeze(0)
        elif ten.ndim == 3 and ten.shape[2] == 1:
            ten = ten.transpose(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError("RCANWrapper expects single-channel inputs")
        return ten.to(device)

    @staticmethod
    def _to_numpy(ten) -> np.ndarray:
        import torch

        out = ten.detach().float().cpu().squeeze().numpy()
        return out.astype(np.float32)

    def infer(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RCAN model not loaded")
        import torch

        x = self._to_tensor(image, self.device)
        with torch.no_grad():
            y = self.model(x)
        return self._to_numpy(y)


