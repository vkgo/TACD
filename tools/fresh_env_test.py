"""Smoke test of the HF repo from a clean environment.

Loads <repo> with AutoModel(trust_remote_code=True) exactly as a user would (text encoders are
downloaded into whatever HF_HOME is set), generates fixed-seed samples and checks
  * shapes and finiteness of every output;
  * rigid skeleton: every bone length is constant over time (FK sanity);
  * optional: max |latent - reference| against a .npz written by another environment.

usage:
  python fresh_env_test.py --repo <dir or id> --out result.npz [--ref other.npz]
"""
import argparse
import json
import sys

import numpy as np
import torch

PROMPTS = [
    "a person walks forward, turns around and walks back",
    "a person jumps twice with arms relaxed at the sides",
    "a person kneels down onto all fours and crawls to the right",
]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="")
    args = ap.parse_args()

    from transformers import AutoModel
    import transformers
    model = AutoModel.from_pretrained(args.repo, trust_remote_code=True).to("cuda").eval()
    out = model.generate(PROMPTS, num_frames=120, seed=[0, 1, 2])
    lat = out.latent.float().cpu().numpy()
    res = {"torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0)}
    shapes = {"latent": lat.shape, "motion": out.motion.shape, "joints": out.joints.shape,
              "rot6d": out.rot6d.shape, "transl": out.transl.shape}
    assert shapes == {"latent": (3, 120, 201), "motion": (3, 120, 201), "joints": (3, 120, 22, 3),
                      "rot6d": (3, 120, 22, 6), "transl": (3, 120, 3)}, shapes
    assert all(np.isfinite(x).all() for x in (lat, out.motion, out.joints, out.rot6d, out.transl))
    j = out.joints
    bones = np.stack([np.linalg.norm(j[:, :, i] - j[:, :, p], axis=-1) for i, p in enumerate(PARENTS) if p >= 0], -1)
    res["bone_len_max_std"] = float(bones.std(axis=1).max())
    assert res["bone_len_max_std"] < 1e-4, res["bone_len_max_std"]
    res["root_path_len_m"] = [float(np.linalg.norm(np.diff(out.transl[b][:, [0, 2]], axis=0), axis=-1).sum()) for b in range(3)]
    res["height_range_m"] = [float(j[b, :, :, 1].max() - j[b, :, :, 1].min()) for b in range(3)]
    # same call twice -> identical (batch composition is kept: fp32 GEMMs depend on batch size)
    again = model.generate(PROMPTS, num_frames=120, seed=[0, 1, 2]).latent.float().cpu().numpy()
    res["repeat_identical"] = bool(np.array_equal(again, lat))
    if args.ref:
        ref = np.load(args.ref)["latent"]
        res["max_abs_vs_ref"] = float(np.abs(ref - lat).max())
        res["mean_abs_vs_ref"] = float(np.abs(ref - lat).mean())
    np.savez(args.out, latent=lat, joints=out.joints)
    print("FRESH_RESULT", json.dumps(res))
    print("FRESH_OK")


if __name__ == "__main__":
    sys.exit(main())
