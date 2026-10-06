"""HY smoothing and WoodenMesh FK with exactly one world-ground translation."""
import sys
import numpy as np
from .motion import HumanMotion
from .paths import hy_motion_root


class HYPostprocessor:
    def __init__(self, device="cpu"):
        import torch
        sys.path.insert(0, str(hy_motion_root()))
        from hymotion.pipeline.body_model import WoodenMesh
        from hymotion.pipeline.motion_diffusion import MotionGeneration
        self.model = WoodenMesh(str(hy_motion_root() / "scripts/gradio/static/assets/dump_wooden")).to(device)
        self.smoothing = MotionGeneration
        self.device = torch.device(device)

    def __call__(self, motion, metadata=None):
        import torch
        raw = torch.as_tensor(motion, dtype=torch.float32, device=self.device)
        if raw.ndim != 2 or raw.shape[1] != 201 or not 11 <= len(raw) <= 360:
            raise ValueError("HY postprocessing expects (T,201), 11<=T<=360 (Savgol window=11)")
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite generated motion")
        with torch.inference_mode():
            rot = self.smoothing.smooth_with_slerp(raw[:, 3:135].reshape(1, -1, 22, 6), sigma=1.0)[0]
            trans = self.smoothing.smooth_with_savgol(raw[:, :3], window_length=11, polyorder=5)
            joints, minima = [], []
            # Chunk FK to avoid retaining the full skinned mesh for long clips.
            for start in range(0, len(raw), 24):
                end = start + 24
                decoded = self.model({"rot6d": rot[start:end], "trans": trans[start:end]})
                joints.append(decoded["keypoints3d"][:, :22])
                minima.append(decoded["vertices"][..., 1].min())
            ground = torch.stack(minima).min()
            aligned = trans.clone()
            aligned[:, 1] -= ground
            # WoodenMesh joints have NO translation. Add aligned translation once.
            world = torch.cat(joints) + aligned[:, None, :]
        result = HumanMotion(rot.cpu().numpy(), aligned.cpu().numpy(), world.cpu().numpy(),
                             ground_offset=float(ground), metadata=metadata or {})
        result.validate()
        return result
