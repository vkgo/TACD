"""Joint tracking errors against the reference, evaluated at 50 Hz before the first reset."""
import json
from pathlib import Path
import numpy as np

from .export import mappings
from .motion import RobotMotion
from .render import sample_qpos
from .transition import generated_time, generated_source_qpos


def generated_tracking_metrics(executed, reference, fps=50., times=None):
    """Evaluate only physical states at/after the generated segment's start."""
    from scipy.spatial.transform import Rotation
    q = np.asarray(executed)
    times = np.arange(len(q)) / fps if times is None else np.asarray(times)
    prefix = float(reference.metadata.get("stand_prefix_seconds", 0.))
    keep = times >= prefix - 1e-9
    if "generated_end_seconds" in reference.metadata:
        keep &= times <= reference.metadata["generated_end_seconds"]+1e-9
    selected = q[keep]
    source_times = generated_time(times[keep]-prefix, reference.metadata.get("generated_time_mapping"))
    common = dict(stand_prefix_seconds=prefix, generated_reached=bool(keep.any()),
                  generated_tracked_states=int(keep.sum()), full_reference_states=len(q),
                  generated_tracked_seconds=max(0., float(times[keep][-1])-prefix) if keep.any() else 0.,
                  generated_source_seconds=float(source_times[-1]) if len(source_times) else 0.)
    if not len(selected):
        fields = [name + "_rmse_rad" for name in ["joint", "legs", "waist", "arms"]]
        return {**dict.fromkeys(fields), "per_joint_rmse_rad": None, "tracked_states": 0,
                "drift_xy_m": None, "root_position_rmse_m": None,
                "root_orientation_rmse_rad": None, **common}
    target = sample_qpos(generated_source_qpos(reference), reference.fps, source_times)
    angle = (Rotation.from_quat(target[:, [4, 5, 6, 3]]).inv() *
             Rotation.from_quat(selected[:, [4, 5, 6, 3]])).magnitude()
    return dict(joint_tracking_metrics(selected, target), **common,
                drift_xy_m=float(np.linalg.norm(selected[-1, :2] - target[-1, :2])),
                root_position_rmse_m=float(np.sqrt(np.mean((selected[:, :3] - target[:, :3])**2))),
                root_orientation_rmse_rad=float(np.sqrt(np.mean(angle**2))))


def failure_phase(failed, prefix_seconds, failure_seconds, time_mapping=None):
    in_prefix = bool(failed and failure_seconds < prefix_seconds-1e-9)
    generated_seconds = (float(generated_time(failure_seconds-prefix_seconds, time_mapping))
                         if failed and not in_prefix else None)
    return dict(failure_seconds=failure_seconds if failed else None,
                failure_generated_seconds=generated_seconds, fail_in_prefix=in_prefix,
                fail_in_generated_first_second=bool(generated_seconds is not None and generated_seconds < 1.-1e-9))


def suffix_outcomes(reference, failure_seconds):
    end = reference.metadata.get("generated_end_seconds", (len(reference.qpos)-1)/reference.fps)
    reference_end = (len(reference.qpos)-1)/reference.fps
    generated_success = failure_seconds is None or failure_seconds > end+1e-9
    prefix = reference.metadata.get("stand_prefix_seconds", 0.)
    in_generated = failure_seconds is not None and prefix-1e-9 <= failure_seconds <= end+1e-9
    return dict(generated_success=bool(generated_success),
        fail_in_generated_first_second=bool(in_generated and float(generated_time(
            failure_seconds-prefix, reference.metadata.get("generated_time_mapping"))) < 1.-1e-9),
        fail_in_suffix=bool(failure_seconds is not None and end+1e-9 < failure_seconds <= reference_end+1e-9
                           and reference.metadata.get("stand_suffix_frames", 0)>0),
        fail_after_reference=bool(failure_seconds is not None and failure_seconds > reference_end+1e-9),
        generated_end_seconds=end, reference_end_seconds=reference_end)


def standing_window(qpos, times, end_seconds, window_seconds=.5):
    """Every state in the closed final window must meet all four criteria."""
    import mujoco
    from scipy.spatial.transform import Rotation
    from .transition import box_feet_model, _foot_box_ids
    from .geometry import geom_floor
    from .vendor.sonic_singleprocess import DEFAULT_ANGLES_MUJOCO
    times = np.asarray(times)
    q = np.asarray(qpos)[(times >= end_seconds-window_seconds-1e-9)&(times <= end_seconds+1e-9)]
    complete = bool(len(times) and times[0]<=end_seconds-window_seconds+1e-9 and times[-1]>=end_seconds-1e-9)
    if not len(q):
        return dict(stood=False, complete=False, states=0, window_end_seconds=end_seconds)
    angles = Rotation.from_quat(q[:, [4,5,6,3]]).as_euler("xyz")[:, :2]
    model=box_feet_model(); data=mujoco.MjData(model); ids=_foot_box_ids(model)
    soles=[]
    for pose in q:
        data.qpos[:]=pose; mujoco.mj_kinematics(model,data)
        soles.append([geom_floor(model,data,g) for g in ids])
    joint=np.sqrt(np.mean((q[:,7:]-DEFAULT_ANGLES_MUJOCO)**2,axis=1))
    flags=dict(pelvis=bool(np.all(q[:,2]>.6)), tilt=bool(np.all(np.abs(angles)<np.deg2rad(15))),
        both_feet=bool(np.all(np.asarray(soles)<.03)), joints=bool(np.all(joint<.15)))
    return dict(stood=bool(complete and all(flags.values())), complete=complete, states=len(q),
        window_end_seconds=end_seconds, window_seconds=window_seconds, checks=flags,
        min_pelvis_z_m=float(q[:,2].min()), max_abs_roll_pitch_rad=float(np.abs(angles).max()),
        max_foot_sole_m=float(np.max(soles)), max_default_joint_rmse_rad=float(joint.max()))


def joint_tracking_metrics(executed, target):
    error = np.asarray(executed, np.float64)[:, 7:] - np.asarray(target, np.float64)[:, 7:]
    result = {name + "_rmse_rad": float(np.sqrt(np.mean(error[:, group] ** 2)))
              for name, group in [("joint", slice(None)), ("legs", slice(0, 12)),
                                  ("waist", slice(12, 15)), ("arms", slice(15, 29))]}
    result["per_joint_rmse_rad"] = np.sqrt(np.mean(error ** 2, axis=0)).tolist()
    result["tracked_states"] = len(error)
    return result


def isaac_trajectories(trajectory_path, references):
    """Extract first episodes by dataset ID, with terminal states and no reset tails."""
    with np.load(trajectory_path, allow_pickle=False) as d:
        ids = d["dataset_motion_ids"]
        keys = d["dataset_motion_keys"].astype(str)
        pre, post = d["pre_step_qpos"], d["qpos"][1:]
        dones, steps, fps = d["dones"], d["control_step_indices"], float(d["fps"])
    order = mappings()["G1_ISAACLAB_TO_MUJOCO_DOF"]
    result = {}
    for name, reference in references.items():
        dataset_id = int(np.flatnonzero(keys == name)[0])
        hits = np.argwhere(ids == dataset_id)
        if not len(hits):
            raise ValueError(f"No recorded states for dataset motion {name}")
        start, env_id = map(int, hits[0])
        stop = start
        # First continuous episode only. Later reset/replay states are excluded.
        while stop < len(ids) and ids[stop, env_id] == dataset_id:
            if stop > start and steps[stop] <= steps[stop - 1]:
                break
            stop += 1
            if dones[stop - 1, env_id]:
                break
        robot = RobotMotion.load(reference)
        # Cap at the nominal reference window when shorter successful clips
        # continue stepping while longer batch peers finish.
        length = min(stop - start, int(np.ceil((len(robot.qpos) - 1) / robot.fps * fps)))
        q = np.concatenate([pre[start:start + 1, env_id], post[start:start + length, env_id]]).astype(np.float64)
        q[:, 7:] = q[:, 7:][:, order]
        times = (steps[start] + np.arange(len(q))) / fps
        result[name] = dict(qpos=q, times=times, fps=fps, reference=robot,
                            dataset_motion_id=dataset_id, env_id=env_id,
                            first_control_step=int(steps[start]))
    return result


def isaac_joint_metrics(trajectory_path, references):
    """Use recorded dataset IDs and local control steps, never file/name order."""
    result = {}
    for name, episode in isaac_trajectories(trajectory_path, references).items():
        q, fps = episode["qpos"], episode["fps"]
        row = generated_tracking_metrics(q, episode["reference"], fps, episode["times"])
        row.update({key: episode[key] for key in ["dataset_motion_id", "env_id", "first_control_step"]})
        row.update(tracked_seconds=(len(q) - 1) / fps,
                   window="Generated segment only, through first terminal state or nominal reference end; no reset tails")
        result[name] = row
    return result
