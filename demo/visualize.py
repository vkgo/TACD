"""Generate motions with TACD and write them to a standalone 3D viewer (one HTML file).

    python demo/visualize.py "a person walks forward, turns around and waves" --duration 5 --seeds 0,1 --out walk.html

From Python, for motions you generated yourself:

    import sys; sys.path.insert(0, "demo")
    from visualize import save_html
    out = model.generate(["a person jumps twice"], duration=4.0, seed=[0])
    save_html(out, "a person jumps twice", "jump.html")

The page draws the wooden character from HY-Motion with three.js; the browser loads three.js and the
character model from public CDNs, so viewing needs an internet connection.
"""
import argparse
import os
import sys
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hy_viewer.utils.visualize_mesh_web import generate_static_html_content, save_visualization_data  # noqa: E402

MODEL_ID = "weijinhuang/TACD-HY-Motion-Lite"


def motion_html(out, text: str, folder: str = "output/viewer") -> str:
    """HTML page (string) showing every motion in `out` (a TACDMotionOutput) side by side."""
    data = {"rot6d": torch.as_tensor(out.rot6d).float().cpu(), "transl": torch.as_tensor(out.transl).float().cpu()}
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # the viewer helpers keep one NPZ per motion under demo/<folder> and read them back
    save_visualization_data(output=data, text=text, rewritten_text=text, timestamp=name,
                            output_dir=os.path.join(HERE, folder), output_filename=name)
    return generate_static_html_content(folder_name=folder, file_name=name, hide_captions=False)


def save_html(out, text: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(motion_html(out, text))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt")
    ap.add_argument("--duration", type=float, default=4.0, help="seconds, at most 12")
    ap.add_argument("--seeds", default="0", help="comma-separated, one motion per seed")
    ap.add_argument("--out", default="motion.html")
    ap.add_argument("--model", default=MODEL_ID, help="Hugging Face id or local copy of the model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()
    with torch.no_grad():
        out = model.generate([args.prompt] * len(seeds), duration=args.duration, seed=seeds)
    print(f"wrote {save_html(out, args.prompt, args.out)}")


if __name__ == "__main__":
    main()
