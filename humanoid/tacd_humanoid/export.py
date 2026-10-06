"""MuJoCo WXYZ qpos -> SONIC XYZW motion-lib and official 50 Hz deploy CSV."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .motion import RobotMotion
from .paths import isaaclab_python, runtime_env, sonic_root


def mappings():
    source = sonic_root() / "gear_sonic/envs/manager_env/robots/g1.py"
    names = {"G1_ISAACLAB_TO_MUJOCO_DOF", "G1_MUJOCO_TO_ISAACLAB_DOF", "G1_ISAACLAB_JOINTS"}
    result = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != names:
        raise ValueError("Upstream G1 joint mapping definitions missing")
    return result


EXPORT_PROTOCOL = "exact_source_grid_50hz_v1"


def canonical_qpos(robot, target_fps=50):
    """Exclusive endpoint, exact rational indices independent of sequence length."""
    source_fps = int(robot.fps)
    count = ((len(robot.qpos)-1)*target_fps + source_fps-1)//source_fps
    numerators = np.arange(count, dtype=np.int64)*source_fps
    left, remainder = np.divmod(numerators, target_fps)
    right = np.minimum(left+1, len(robot.qpos)-1)
    weight = remainder.astype(np.float64)/target_fps
    q = np.asarray(robot.qpos, np.float64)
    result = q[left]+(q[right]-q[left])*weight[:, None]
    # SciPy uses local interval differences; integer source samples are restored
    # exactly, avoiding the upstream small-angle midpoint branch entirely.
    result[:, 3:7] = Slerp(np.arange(len(q), dtype=float),
        Rotation.from_quat(q[:, [4,5,6,3]]))(left+weight).as_quat()[:, [3,0,1,2]]
    result[remainder == 0] = q[left[remainder == 0]]
    return result


def to_motionlib(robot, out, name="motion"):
    robot.validate()
    source = sonic_root() / "gear_sonic/data_process/convert_soma_csv_to_motion_lib.py"
    spec = importlib.util.spec_from_file_location("sonic_motion_converter", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    q = canonical_qpos(robot).astype(np.float32)
    root_xyzw = q[:, [4, 5, 6, 3]]
    dof = q[:, 7:]
    aa = np.zeros((len(q), 30, 3), np.float32)
    aa[:, 0] = Rotation.from_quat(root_xyzw).as_rotvec()
    aa[:, 1:] = module.DOF_AXIS[None] * dof[..., None]
    data = {name: dict(root_trans_offset=q[:, :3], root_rot=root_xyzw, dof=dof,
                      pose_aa=aa, smpl_joints=np.zeros((len(q), 24, 3), np.float32), fps=50)}
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    file = out / f"{name}.pkl"
    joblib.dump(data, file)
    return file


def to_sonic(robot, out, name="motion"):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    motionlib = to_motionlib(robot, out / "motionlib", name)
    command = [isaaclab_python(),
        str(sonic_root() / "gear_sonic/data_process/export_motion_to_deploy.py"),
        str(motionlib), "--target-fps", "50", "--output-dir", str(out / "csv")]
    with (out / "export.log").open("w") as log:
        subprocess.run(command, cwd=sonic_root(), env=runtime_env(out, isolate_home=True), stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=300)
    reference = dict(motionlib=str(motionlib), csv=str(out / "csv" / name), source_fps=30, target_fps=50,
                     export_protocol=EXPORT_PROTOCOL, motionlib_fps=50,
                     motionlib_joint_order="mujoco", motionlib_quat="xyzw", csv_joint_order="isaaclab", csv_quat="wxyz")
    (out / "reference.json").write_text(json.dumps(reference, indent=2))
    validation = validate_csv(robot, reference["csv"])
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    return reference


def validate_csv(robot, directory):
    from .render import sample_qpos
    directory = Path(directory)
    joints = np.loadtxt(directory / "joint_pos.csv", delimiter=",", skiprows=1)
    pos = np.loadtxt(directory / "body_pos.csv", delimiter=",", skiprows=1).reshape(-1, 14, 3)
    quat = np.loadtxt(directory / "body_quat.csv", delimiter=",", skiprows=1).reshape(-1, 14, 4)
    times = np.arange(0, (len(robot.qpos) - 1) / robot.fps, 1 / 50.)
    assert joints.shape == (len(times), 29) and len(pos) == len(times)
    assert np.isfinite(joints).all() and np.isfinite(pos).all() and np.isfinite(quat).all()
    expected = sample_qpos(robot.qpos, robot.fps, times)
    joint_error = float(np.abs(joints - expected[:, 7:][:, mappings()["G1_MUJOCO_TO_ISAACLAB_DOF"]]).max())
    root_position_error = float(np.abs(pos[:, 0] - expected[:, :3]).max())
    root_rotation_error = float((Rotation.from_quat(quat[:, 0, [1, 2, 3, 0]]).inv() *
                                 Rotation.from_quat(expected[:, [4, 5, 6, 3]])).magnitude().max())
    assert joint_error < 2e-5 and root_position_error < 2e-5 and root_rotation_error < 2e-5
    return dict(passed=True, frames=len(times), fps=50, joint_order="isaaclab", quaternion_order="wxyz",
                joint_max_error_rad=joint_error, root_position_max_error_m=root_position_error,
                root_rotation_max_error_rad=root_rotation_error,
                export_protocol=EXPORT_PROTOCOL,
                note="Exact source grid in wrapper; official FK/export at native 50 Hz; CSV precision 6 decimals.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qpos", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(to_sonic(RobotMotion.load(args.qpos), args.out))
