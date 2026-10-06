"""TACD (8 steps, guidance 5) text-to-motion with HY-Motion postprocessing."""
import argparse
import json
import os
from pathlib import Path
import time

from .human import HYPostprocessor
from .paths import tacd_model


class MotionGenerator:
    def __init__(self, device="cuda:0"):
        import torch
        self.device = device
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        started = time.perf_counter()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(tacd_model(), trust_remote_code=True).to(device).eval()
        self.postprocess = HYPostprocessor(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    def generate(self, text, duration=4.0, seed=0):
        import torch
        frames = round(duration * 30)
        if not 20 <= frames <= 360:
            raise ValueError("Duration must produce 20..360 frames at 30 fps")
        times = {"model_load": self.load_seconds}
        started = time.perf_counter()
        with torch.inference_mode():
            # Decompose timing without changing the HF generate() sampling path.
            noise = self.model.make_noise(frames, [seed]).to(self.device, dtype=self.model.dtype)
            text_start = time.perf_counter()
            cond = self.model.encode_text([text])
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            times["text_encode"] = time.perf_counter() - text_start
            sample_start = time.perf_counter()
            latent = self.model.sample(noise, cond, 8, 5.0)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            times["motion_sample"] = time.perf_counter() - sample_start
            raw = self.model.decode(latent)[0][0]
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            times["generation"] = time.perf_counter() - started
            post_start = time.perf_counter()
            human = self.postprocess(raw, dict(text=text, seed=seed, backend="tacd",
                steps=8, guidance_scale=5.0,
                postprocessing="HY SLERP sigma1 / Savgol 11,5 / Wooden FK / single ground alignment"))
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            times["postprocess"] = time.perf_counter() - post_start
            human.metadata["latency_seconds"] = times
        return human


def generate(text, duration=4.0, seed=0, device="cuda:0"):
    return MotionGenerator(device).generate(text, duration, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    human = generate(a.text, a.duration, a.seed, a.device)
    human.save(out / "human.npz")
    (out / "generation.json").write_text(json.dumps(human.metadata, indent=2))
    print(json.dumps(human.metadata, indent=2))


if __name__ == "__main__":
    main()
