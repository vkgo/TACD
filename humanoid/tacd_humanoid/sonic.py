"""Track motion-lib clips with SONIC v1.1 in IsaacLab through the official evaluator."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from .paths import isaaclab_python, runtime_env, sonic_checkpoint, sonic_root


def track(motions, out, num_envs=1, record=True, encoder="g1", initial_joint_noise_rad=0.0, perturb_seed=0, motion_order=None):
    if encoder not in ["g1", "teleop", "mixed"]:
        raise ValueError("encoder must be g1, teleop, or mixed (historical reproduction)")
    import math
    if not math.isfinite(initial_joint_noise_rad) or initial_joint_noise_rad < 0:
        raise ValueError("initial_joint_noise_rad must be finite and nonnegative")
    if perturb_seed < 0:
        raise ValueError("perturb_seed must be nonnegative")
    motions, out = Path(motions).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = list(motions.glob("*.pkl"))
    if not files:
        raise ValueError(f"No motion-lib files in {motions}")
    checkpoint = sonic_checkpoint()
    python = isaaclab_python()
    env = runtime_env(out, isolate_home=True)
    env["TACD_CHECKPOINT"] = str(checkpoint)
    env["TACD_RECORD_TRAJECTORY"] = "1" if record else "0"
    env["TACD_ENCODER"] = encoder
    env["TACD_INITIAL_JOINT_NOISE_RAD"] = str(initial_joint_noise_rad)
    env["TACD_PERTURB_SEED"] = str(perturb_seed)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = os.path.dirname(python) + ":" + env["PATH"]
    for name in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE"]:
        env.pop(name, None)
    command = [
        python, "-u", "gear_sonic/eval_agent_trl.py",
        f"+checkpoint={checkpoint}", "+headless=True", "++eval_callbacks=im_eval",
        "++callbacks.im_eval._target_=tacd_humanoid.policy_evidence.PolicyEvidenceCallback",
        "++run_eval_loop=False", f"++num_envs={num_envs}", f"++eval_output_dir={out}",
        f"hydra.run.dir={out}/hydra", f"++output_dir={out}",
        f"++manager_env.config.save_rendering_dir={out}/renderings",
        "++manager_env.observations.policy.enable_corruption=False",
        "++manager_env.observations.tokenizer.enable_corruption=False",
        "+manager_env/terminations=tracking/eval",
        f"++manager_env.commands.motion.motion_lib_cfg.max_unique_motions={len(files) + 8}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motions}",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
    ]
    if motion_order is not None:
        order = json.loads(Path(motion_order).read_text())
        if len(order) != len(files) or set(order) != {p.stem for p in files}:
            raise ValueError("motion_order must contain every motion key exactly once")
        # Upstream exact-key filtering preserves supplied order before assigning
        # motions to environment slots. Avoid filesystem enumeration as a seed.
        command.append("++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=" + json.dumps(order))
    if encoder != "mixed":
        command.append(f"+use_encoder={encoder}")
    started = time.time()
    (out / "command.json").write_text(json.dumps(dict(argv=command, cwd=str(sonic_root()),
        started_unix=started, cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
        runtime_tmp=env["TMPDIR"], initial_joint_noise_rad=initial_joint_noise_rad, perturb_seed=perturb_seed), indent=2))
    with (out / "eval.log").open("w") as log:
        subprocess.run(command, cwd=sonic_root(), env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=3600)
    metrics = json.loads((out / "metrics_eval.json").read_text())
    evidence = json.loads((out / "policy_evidence.json").read_text())
    if evidence["policy_calls"] < 1 or evidence["physics_steps"] < 1 or evidence["action_abs_max"] == 0:
        raise RuntimeError("No evidence of nonzero policy actions applied to physics")
    return metrics, dict(evidence, wall_seconds=time.time() - started,
                         startup_to_first_command_seconds=evidence["first_command_unix"] - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--encoder", choices=["g1", "teleop", "mixed"], default="g1")
    parser.add_argument("--initial-joint-noise-rad", type=float, default=0.0)
    parser.add_argument("--perturb-seed", type=int, default=0)
    parser.add_argument("--motion-order", help="JSON list of motion keys in environment-slot order")
    args = parser.parse_args()
    metrics, evidence = track(args.motions, args.out, args.num_envs, not args.no_record, args.encoder, args.initial_joint_noise_rad, args.perturb_seed, args.motion_order)
    print(json.dumps({"success_rate": metrics["eval/success/success_rate"], **evidence}, indent=2))


if __name__ == "__main__":
    main()
