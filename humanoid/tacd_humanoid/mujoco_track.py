"""SONIC v1.1 G1-mode ONNX policy controlling real MuJoCo dynamics."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import RobotMotion
from .paths import runtime_env


def track_mujoco(csv_path, out, reference, *, initial_joint_noise_rad=0.0, seed=0, render_video=True,
                 duration_seconds=None, post_motion_behavior="decode"):
    """Start the ONNX worker with its environment's CUDA library search path.

    glibc reads LD_LIBRARY_PATH at process start. The inherited cluster path can
    point at a different Python/CUDA installation, so setting it after importing
    torch in the generator process cannot fix cuDNN's lazy sublibrary lookup.
    """
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    reference_path = out / "mujoco_reference.npz"
    reference.save(reference_path)
    env = runtime_env(out)
    libraries = []
    for name in ["cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "curand", "cufft"]:
        spec = importlib.util.find_spec("nvidia." + name)
        if spec is not None:
            libraries.append(str(Path(next(iter(spec.submodule_search_locations))) / "lib"))
    env["LD_LIBRARY_PATH"] = ":".join(libraries)
    worker = (
        "import sys, json; from tacd_humanoid.mujoco_track import _track_mujoco_in_process; "
        "from tacd_humanoid.motion import RobotMotion; "
        "_track_mujoco_in_process(sys.argv[1], sys.argv[2], RobotMotion.load(sys.argv[3]), **json.loads(sys.argv[4]))"
    )
    command = [sys.executable, "-u", "-c", worker, str(Path(csv_path).resolve()),
               str(out), str(reference_path), json.dumps(dict(
                   initial_joint_noise_rad=initial_joint_noise_rad, seed=seed, render_video=render_video,
                   duration_seconds=duration_seconds, post_motion_behavior=post_motion_behavior))]
    (out / "mujoco_command.json").write_text(json.dumps(
        dict(argv=command, cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
             library_path=libraries), indent=2))
    with (out / "mujoco.log").open("w") as log:
        subprocess.run(command, env=env, check=True, timeout=900,
                       stdout=log, stderr=subprocess.STDOUT)
    return (json.loads((out / "mujoco_metrics.json").read_text()),
            json.loads((out / "policy_evidence.json").read_text()))


def _track_mujoco_in_process(csv_path, out, reference, *, initial_joint_noise_rad=0.0, seed=0, render_video=True,
                             duration_seconds=None, post_motion_behavior="decode"):
    if post_motion_behavior not in ("decode", "deploy"):
        raise ValueError("post_motion_behavior must be decode or deploy")
    if initial_joint_noise_rad < 0:
        raise ValueError("initial_joint_noise_rad must be nonnegative")
    if duration_seconds is not None and (not np.isfinite(duration_seconds) or duration_seconds<=0):
        raise ValueError("duration_seconds must be finite and positive")
    if duration_seconds is not None and not np.isclose(
            duration_seconds * 50, round(duration_seconds * 50), rtol=0, atol=1e-9):
        raise ValueError("duration_seconds must be a multiple of the 0.02 s control interval")
    from .vendor import sonic_singleprocess as sp
    from .render import render_comparison, sample_qpos
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    motion = sp._load_motion_sequence(Path(csv_path), "motion-data", "xyzw", "wxyz",
        "isaaclab", 50., 50., 0)
    dec = sp._make_ort_session(sp.DEFAULT_DECODER_ONNX)
    enc = sp._make_ort_session(sp.DEFAULT_ENCODER_ONNX)
    specs, dim, especs, modes = sp._build_observation_specs(sp.DEFAULT_OBS_CONFIG,
        int(dec.get_inputs()[0].shape[1]), int(enc.get_inputs()[0].shape[1]))
    player = sp.SonicSingleProcessPlayer(motion=motion, xml_path=sp.DEFAULT_XML,
        decoder_session=dec, encoder_session=enc, planner_runner=None,
        decoder_obs_specs=specs, token_dim=dim, encoder_obs_specs=especs, encoder_modes=modes,
        control_decimation=4, sim_dt=.005, warmup_steps=0, loop_motion=False,
        planner_enabled=False, planner_drive_motion=False, planner_replan_interval=1.,
        movement_state=sp.MovementState(0, np.zeros(3), np.array([1., 0, 0]), 0., .75, 0),
        post_motion_behavior=post_motion_behavior, print_every_seconds=30., save_p0_trace=True)
    if initial_joint_noise_rad:
        player.data.qpos[7:36] += np.random.default_rng(seed).uniform(
            -initial_joint_noise_rad, initial_joint_noise_rad, 29)
        sp.mujoco.mj_forward(player.model, player.data)
        player.history.clear()
        for _ in range(120):
            player._append_robot_state()
    initial_qpos = player.data.qpos.copy()
    initial_qvel = player.data.qvel.copy()
    expected_history_q = initial_qpos[7:][sp.MUJOCO_TO_ISAACLAB] - sp.DEFAULT_ANGLES_ISAACLAB
    expected_history_dq = initial_qvel[6:][sp.MUJOCO_TO_ISAACLAB]
    history_error = max(float(np.max(np.abs(entry.body_q_isaac - expected_history_q)))
                        for entry in player.history)
    history_velocity_error = max(float(np.max(np.abs(entry.body_dq_isaac - expected_history_dq)))
                                 for entry in player.history)
    initial_history_samples = len(player.history)
    summary = player.run(None, 25, 640, 480, 3., 135., -18.,
                         0 if duration_seconds is None else duration_seconds, duration_seconds is None)
    executed = player.rollout_qpos
    np.savez_compressed(out / "trajectory.npz", qpos=executed, qvel=player.rollout_qvel,
        fps=50., joint_order="mujoco", quaternion_order="wxyz", source="MuJoCo SONIC v1.1 policy",
        timing="Initial state then 50 Hz states through the final complete control interval")
    np.savez_compressed(out / "policy_actions.npz", actions=player.p0_action_isaac_hist,
        targets=player.p0_target_q_mujoco_hist, torques=player.p0_ctrl_hist)
    evidence = dict(controller="SONIC v1.1", simulator="MuJoCo", kinematic_replay=False,
        policy_calls=player.policy_calls, physics_steps=len(player.p0_ctrl_hist),
        control_steps=summary["control_steps"], control_decimation=player.control_decimation,
        physics_dt_seconds=player.model.opt.timestep, trajectory_states=len(executed),
        action_abs_max=float(np.abs(player.p0_action_isaac_hist).max()),
        first_command_unix=player.first_command_unix, wall_seconds=time.time() - started,
        encoder_providers=enc.get_providers(), decoder_providers=dec.get_providers(),
        orientation_mode=1, encoder_mode=0, encoder="g1", ankle_pitch_kp_kd_scale=1.0,
        initial_joint_noise_rad=initial_joint_noise_rad, seed=seed, history_initialization="120 current-state samples",
        initial_qpos=initial_qpos.tolist(), initial_qvel=initial_qvel.tolist(),
        initial_history_samples=initial_history_samples,
        initial_history_joint_max_error=history_error,
        initial_history_joint_velocity_max_error=history_velocity_error,
        post_motion_behavior=post_motion_behavior,requested_duration_seconds=duration_seconds,
        simulated_seconds=summary["simulated_seconds"],reference_control_frames=motion.timesteps,
        final_reference_frame=summary["final_frame"],reference_finished=summary["motion_finished"],
        armature_source="gear_sonic/envs/manager_env/robots/g1.py:G1_CYLINDER_MODEL_12_DEX_CFG",
        armature_mujoco=player.model.dof_armature[6:].tolist(),
        joint_damping=0.0, joint_frictionloss=0.0, foot_collision="official G1 box")
    if evidence["policy_calls"] < 1 or evidence["action_abs_max"] == 0:
        raise RuntimeError("Policy did not generate nonzero control actions")
    video = render_comparison(reference, executed, 50., out / "video.mp4", post_motion_behavior=post_motion_behavior) if render_video else None
    from .tracking_metrics import generated_tracking_metrics, failure_phase
    falls = np.flatnonzero(executed[:, 2] < .35)
    result = dict(video=video, motion_finished=summary["motion_finished"],
        min_root_height_m=float(executed[:, 2].min()),
        fell_below_035m=bool(len(falls)),
        metric_note="Generated-segment errors only; fall detection covers the full reference. Not IsaacLab MPJPE/success.")
    result.update(generated_tracking_metrics(executed, reference))
    result.update(failure_phase(bool(len(falls)), reference.metadata.get("stand_prefix_seconds", 0.),
                               float(falls[0]) / 50 if len(falls) else None,
                               reference.metadata.get("generated_time_mapping")))
    from .tracking_metrics import suffix_outcomes, standing_window
    result.update(suffix_outcomes(reference, float(falls[0])/50 if len(falls) else None))
    result["ending_standing"] = standing_window(executed, np.arange(len(executed))/50, (len(executed)-1)/50)
    (out / "policy_evidence.json").write_text(json.dumps(evidence, indent=2))
    (out / "mujoco_metrics.json").write_text(json.dumps(result, indent=2))
    return result, evidence


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True); p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--initial-joint-noise-rad", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--duration-seconds",type=float,default=None,
                   help="Optional horizon, a positive multiple of 0.02 s; continue decode after the reference ends")
    p.add_argument("--post-motion-behavior", choices=["decode","deploy"], default="decode")
    a = p.parse_args()
    print(json.dumps(track_mujoco(a.csv, a.out, RobotMotion.load(a.reference),
        initial_joint_noise_rad=a.initial_joint_noise_rad, seed=a.seed, render_video=not a.no_video,
        duration_seconds=a.duration_seconds, post_motion_behavior=a.post_motion_behavior), indent=2))
