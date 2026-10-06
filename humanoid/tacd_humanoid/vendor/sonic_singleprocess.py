#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
#
# Portions ported from NVIDIA GR00T-WholeBodyControl (Apache-2.0),
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES: the policy constants, joint orderings and
# observation layout follow gear_sonic_deploy (policy_parameters.hpp and the C++ observation
# builders). Modified by the TACD authors: rewritten as a single Python process that runs the MuJoCo
# simulation and the SONIC encoder/planner/decoder ONNX models together.
"""Single-process SONIC MuJoCo playback with encoder + planner + decoder.

This runner keeps simulation and policy inference in one Python process:
- MuJoCo simulation loop (no sim subprocess),
- encoder ONNX (token_state),
- planner ONNX (optional, same process),
- decoder ONNX (29-DoF full-body action).

Input supports:
1) `.pkl` with keys: fps, root_pos, root_rot, dof_pos
2) SONIC motion-data clip dir with CSV files:
   - joint_pos.csv / joint_vel.csv
   - body_pos.csv / body_quat.csv
   - optional smpl_joint.csv / smpl_pose.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
import yaml


from ..paths import box_feet_xml, sonic_root

# GR00T-WholeBodyControl checkout with the sonic_v1_1 deploy models (humanoid/sonic/install.sh).
DEFAULT_REPO = sonic_root().expanduser().resolve()
DEFAULT_XML = box_feet_xml()
DEFAULT_POLICY = DEFAULT_REPO / "gear_sonic_deploy" / "policy" / "sonic_v1_1"
DEFAULT_OBS_CONFIG = DEFAULT_POLICY / "observation_config.yaml"
DEFAULT_ENCODER_ONNX = DEFAULT_POLICY / "model_encoder.onnx"
DEFAULT_DECODER_ONNX = DEFAULT_POLICY / "model_decoder.onnx"
DEFAULT_PLANNER_ONNX = DEFAULT_REPO / "gear_sonic_deploy" / "planner" / "target_vel" / "V2" / "planner_sonic.onnx"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _make_ort_session(onnx_path: Path) -> ort.InferenceSession:
    """Prefer CUDA inference, retaining bounded CPU threads for fallback.

    ONNXRuntime's CPU provider otherwise creates large per-process thread pools,
    which makes multi-sample rollout collection oversubscribe heavily.
    """
    intra_threads = max(1, _env_int("SONIC_ORT_INTRA_OP_NUM_THREADS", _env_int("ORT_NUM_THREADS", 1)))
    inter_threads = max(1, _env_int("SONIC_ORT_INTER_OP_NUM_THREADS", _env_int("ORT_INTER_OP_NUM_THREADS", 1)))
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = intra_threads
    sess_options.inter_op_num_threads = inter_threads
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    # Import torch first so its CUDA/cuDNN shared libraries are available to ORT.
    import torch
    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=providers)
    print(f"[ORT] {onnx_path.name}: {session.get_providers()}", flush=True)
    return session


# Mapping arrays from policy_parameters.hpp
# isaaclab_to_mujoco[mujoco_index] -> isaaclab_index
ISAACLAB_TO_MUJOCO = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
# mujoco_to_isaaclab[isaaclab_index] -> mujoco_index
MUJOCO_TO_ISAACLAB = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

LOWER_BODY_ISAACLAB_INDEX = np.asarray([0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18], dtype=np.int64)
WRIST_ISAACLAB_INDEX = np.asarray([23, 24, 25, 26, 27, 28], dtype=np.int64)


def _build_policy_parameters() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build action scales, Kp/Kd, default angles from policy_parameters.hpp."""
    armature_5020 = 0.003609725
    armature_7520_14 = 0.010177520
    armature_7520_22 = 0.025101925
    armature_4010 = 0.00425

    natural_freq = 10.0 * 2.0 * math.pi
    damping_ratio = 2.0

    stiffness_5020 = armature_5020 * natural_freq * natural_freq
    stiffness_7520_14 = armature_7520_14 * natural_freq * natural_freq
    stiffness_7520_22 = armature_7520_22 * natural_freq * natural_freq
    stiffness_4010 = armature_4010 * natural_freq * natural_freq

    damping_5020 = 2.0 * damping_ratio * armature_5020 * natural_freq
    damping_7520_14 = 2.0 * damping_ratio * armature_7520_14 * natural_freq
    damping_7520_22 = 2.0 * damping_ratio * armature_7520_22 * natural_freq
    damping_4010 = 2.0 * damping_ratio * armature_4010 * natural_freq

    effort_5020 = 25.0
    effort_7520_14 = 88.0
    effort_7520_22 = 139.0
    effort_4010 = 5.0

    action_scale = np.asarray(
        [
            0.25 * effort_7520_22 / stiffness_7520_22,
            0.25 * effort_7520_22 / stiffness_7520_22,
            0.25 * effort_7520_14 / stiffness_7520_14,
            0.25 * effort_7520_22 / stiffness_7520_22,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_7520_22 / stiffness_7520_22,
            0.25 * effort_7520_22 / stiffness_7520_22,
            0.25 * effort_7520_14 / stiffness_7520_14,
            0.25 * effort_7520_22 / stiffness_7520_22,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_7520_14 / stiffness_7520_14,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_4010 / stiffness_4010,
            0.25 * effort_4010 / stiffness_4010,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_5020 / stiffness_5020,
            0.25 * effort_4010 / stiffness_4010,
            0.25 * effort_4010 / stiffness_4010,
        ],
        dtype=np.float64,
    )

    kps = np.asarray(
        [
            stiffness_7520_22,
            stiffness_7520_22,
            stiffness_7520_14,
            stiffness_7520_22,
            2.0 * stiffness_5020,
            2.0 * stiffness_5020,
            stiffness_7520_22,
            stiffness_7520_22,
            stiffness_7520_14,
            stiffness_7520_22,
            2.0 * stiffness_5020,
            2.0 * stiffness_5020,
            stiffness_7520_14,
            2.0 * stiffness_5020,
            2.0 * stiffness_5020,
            stiffness_5020,
            stiffness_5020,
            stiffness_5020,
            stiffness_5020,
            stiffness_5020,
            stiffness_4010,
            stiffness_4010,
            stiffness_5020,
            stiffness_5020,
            stiffness_5020,
            stiffness_5020,
            stiffness_5020,
            stiffness_4010,
            stiffness_4010,
        ],
        dtype=np.float64,
    )

    kds = np.asarray(
        [
            damping_7520_22,
            damping_7520_22,
            damping_7520_14,
            damping_7520_22,
            2.0 * damping_5020,
            2.0 * damping_5020,
            damping_7520_22,
            damping_7520_22,
            damping_7520_14,
            damping_7520_22,
            2.0 * damping_5020,
            2.0 * damping_5020,
            damping_7520_14,
            2.0 * damping_5020,
            2.0 * damping_5020,
            damping_5020,
            damping_5020,
            damping_5020,
            damping_5020,
            damping_5020,
            damping_4010,
            damping_4010,
            damping_5020,
            damping_5020,
            damping_5020,
            damping_5020,
            damping_5020,
            damping_4010,
            damping_4010,
        ],
        dtype=np.float64,
    )

    default_angles = np.asarray(
        [
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.363,
            0.0,
            -0.312,
            0.0,
            0.0,
            0.669,
            -0.363,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
            0.0,
            0.6,
            0.0,
            0.0,
            0.0,
            0.2,
            -0.2,
            0.0,
            0.6,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return action_scale, kps, kds, default_angles


G1_ACTION_SCALE, G1_KPS, G1_KDS, DEFAULT_ANGLES_MUJOCO = _build_policy_parameters()
DEFAULT_ANGLES_ISAACLAB = DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAACLAB]


OBS_DIMS: Dict[str, int] = {
    # Decoder/history observations
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
    # Encoder observations used in release config
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_anchor_orientation_heading": 6,
    "motion_anchor_orientation_heading_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "smpl_anchor_orientation_heading_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}


@dataclass
class ObservationSpec:
    name: str
    dim: int
    offset: int


@dataclass
class HeadingState:
    init_base_quat_wxyz: np.ndarray
    delta_heading: float = 0.0


@dataclass
class MovementState:
    locomotion_mode: int
    movement_direction: np.ndarray  # [3]
    facing_direction: np.ndarray  # [3]
    movement_speed: float
    height: float
    random_seed: int


@dataclass
class MotionSequence:
    name: str
    fps: float
    joint_pos_isaac: np.ndarray  # [T, 29]
    joint_vel_isaac: np.ndarray  # [T, 29]
    body_pos: np.ndarray  # [T, 3]
    body_quat_wxyz: np.ndarray  # [T, 4]
    smpl_joints: Optional[np.ndarray] = None  # [T, J, 3]
    smpl_pose: Optional[np.ndarray] = None  # [T, P, 3]
    encode_mode: int = 0
    resolved_joint_order: str = "isaaclab"

    @property
    def timesteps(self) -> int:
        return int(self.joint_pos_isaac.shape[0])

    def ensure_lengths_match(self) -> None:
        n = self.timesteps
        for x, name in [
            (self.joint_vel_isaac, "joint_vel_isaac"),
            (self.body_pos, "body_pos"),
            (self.body_quat_wxyz, "body_quat_wxyz"),
        ]:
            if int(x.shape[0]) != n:
                raise ValueError(f"{self.name}: frame mismatch for {name}: {x.shape[0]} vs {n}")
        if self.smpl_joints is not None and int(self.smpl_joints.shape[0]) != n:
            raise ValueError(f"{self.name}: smpl_joints frame mismatch: {self.smpl_joints.shape[0]} vs {n}")
        if self.smpl_pose is not None and int(self.smpl_pose.shape[0]) != n:
            raise ValueError(f"{self.name}: smpl_pose frame mismatch: {self.smpl_pose.shape[0]} vs {n}")


@dataclass
class RobotStateEntry:
    base_quat_wxyz: np.ndarray  # [4]
    base_ang_vel_body: np.ndarray  # [3]
    body_q_isaac: np.ndarray  # [29]
    body_dq_isaac: np.ndarray  # [29]
    last_action_isaac: np.ndarray  # [29]


def _normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, eps, None)
    return q / n


def _detect_quat_format(q: np.ndarray) -> str:
    q = np.asarray(q, dtype=np.float64)
    m0 = float(np.mean(np.abs(q[:, 0])))
    m3 = float(np.mean(np.abs(q[:, 3])))
    return "wxyz" if m0 >= m3 else "xyzw"


def _to_wxyz(q: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "wxyz":
        return np.asarray(q, dtype=np.float64).copy()
    if fmt == "xyzw":
        return np.asarray(q, dtype=np.float64)[:, [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion format: {fmt}")


def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.asarray([w, x, y, z], dtype=np.float64)


def _quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(np.asarray(q, dtype=np.float64))
    v = np.asarray(v, dtype=np.float64)
    qv = q[..., 1:4]
    qw = q[..., 0:1]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_apply_wxyz(_quat_conjugate_wxyz(q), v)


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _normalize_quat_wxyz(np.asarray(q0, dtype=np.float64))
    q1 = _normalize_quat_wxyz(np.asarray(q1, dtype=np.float64))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat_wxyz((1.0 - t) * q0 + t * q1)
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    return w0 * q0 + w1 * q1


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(np.asarray(q, dtype=np.float64))
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _calc_heading_rad_from_quat(q: np.ndarray) -> float:
    ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    rot = _quat_apply_wxyz(q, ref)
    return float(math.atan2(rot[1], rot[0]))


def _quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / n
    half = 0.5 * angle
    s = math.sin(half)
    c = math.cos(half)
    return _normalize_quat_wxyz(np.asarray([c, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64))


def _calc_heading_quat_wxyz(q: np.ndarray) -> np.ndarray:
    heading = _calc_heading_rad_from_quat(q)
    return _quat_from_angle_axis(heading, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))


def _calc_heading_quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    heading = _calc_heading_rad_from_quat(q)
    return _quat_from_angle_axis(-heading, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))


def _resample_linear(data: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.shape[0] <= 1 or abs(source_fps - target_fps) < 1e-9:
        return data.copy()
    src_t = np.arange(data.shape[0], dtype=np.float64) / source_fps
    dst_t = np.arange(0.0, src_t[-1] + 1e-9, 1.0 / target_fps, dtype=np.float64)
    flat = data.reshape(data.shape[0], -1)
    out = np.empty((len(dst_t), flat.shape[1]), dtype=np.float64)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, flat[:, i])
    return out.reshape((len(dst_t),) + data.shape[1:])


def _resample_quat_nlerp_wxyz(quat: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    quat = _normalize_quat_wxyz(np.asarray(quat, dtype=np.float64))
    if quat.shape[0] <= 1 or abs(source_fps - target_fps) < 1e-9:
        return quat.copy()

    src_t = np.arange(quat.shape[0], dtype=np.float64) / source_fps
    dst_t = np.arange(0.0, src_t[-1] + 1e-9, 1.0 / target_fps, dtype=np.float64)
    out = np.empty((len(dst_t), 4), dtype=np.float64)
    for i, t in enumerate(dst_t):
        pos = t * source_fps
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, quat.shape[0] - 1)
        alpha = float(pos - i0)
        q0 = quat[i0]
        q1 = quat[i1]
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        out[i] = _normalize_quat_wxyz((1.0 - alpha) * q0 + alpha * q1)
    return out


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] <= 1:
        return np.zeros_like(x, dtype=np.float64)
    return np.gradient(x, dt, axis=0)


def _load_pkl(path: Path) -> Dict[str, Any]:
    try:
        import joblib  # type: ignore

        data = joblib.load(path)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)}")
    return data


def _load_csv_matrix(path: Path) -> np.ndarray:
    arr = np.loadtxt(str(path), delimiter=",", skiprows=1, dtype=np.float64)
    return np.atleast_2d(arr)


def _infer_joint_order_auto(joint_pos: np.ndarray) -> str:
    q = np.asarray(joint_pos, dtype=np.float64)
    frames = min(60, q.shape[0])
    q_eval = q[:frames]
    err_mujoco = float(np.mean(np.abs(q_eval - DEFAULT_ANGLES_MUJOCO[None, :])))
    err_isaac = float(np.mean(np.abs(q_eval - DEFAULT_ANGLES_ISAACLAB[None, :])))
    return "mujoco" if err_mujoco <= err_isaac else "isaaclab"


def _convert_joint_order_to_isaac(joint_pos: np.ndarray, joint_order: str) -> np.ndarray:
    jp = np.asarray(joint_pos, dtype=np.float64)
    if joint_order == "isaaclab":
        return jp.copy()
    if joint_order == "mujoco":
        return jp[:, MUJOCO_TO_ISAACLAB]
    raise ValueError(f"Unsupported joint_order: {joint_order}")


def _extract_optional_smpl_pkl(data: Dict[str, Any], fps: float, target_fps: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    smpl_joint_keys = ["smpl_joints", "smpl_joint"]
    smpl_pose_keys = ["smpl_pose", "smpl_poses"]
    smpl_joints = None
    smpl_pose = None

    for k in smpl_joint_keys:
        if k in data:
            arr = np.asarray(data[k], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] % 3 == 0:
                arr = arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)
            if arr.ndim == 3 and arr.shape[2] == 3:
                smpl_joints = _resample_linear(arr, fps, target_fps)
            break

    for k in smpl_pose_keys:
        if k in data:
            arr = np.asarray(data[k], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] % 3 == 0:
                arr = arr.reshape(arr.shape[0], arr.shape[1] // 3, 3)
            if arr.ndim == 3 and arr.shape[2] == 3:
                smpl_pose = _resample_linear(arr, fps, target_fps)
            break

    return smpl_joints, smpl_pose


def _load_motion_from_pkl(
    path: Path,
    root_rot_format: str,
    joint_order: str,
    target_fps: float,
    encode_mode: int,
) -> MotionSequence:
    data = _load_pkl(path)
    for k in ["fps", "root_pos", "root_rot", "dof_pos"]:
        if k not in data:
            raise KeyError(f"{path}: missing key {k}")
    source_fps = float(data["fps"])
    if source_fps <= 0:
        raise ValueError(f"{path}: invalid fps={source_fps}")

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: root_pos shape must be [T,3], got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"{path}: root_rot shape must be [T,4], got {root_rot.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 29:
        raise ValueError(f"{path}: dof_pos shape must be [T,29], got {dof_pos.shape}")

    frames = min(root_pos.shape[0], root_rot.shape[0], dof_pos.shape[0])
    if frames < 2:
        raise ValueError(f"{path}: not enough frames ({frames})")
    root_pos = root_pos[:frames]
    root_rot = root_rot[:frames]
    dof_pos = dof_pos[:frames]

    fmt = root_rot_format
    if fmt == "auto":
        fmt = _detect_quat_format(root_rot)
    body_quat_wxyz = _normalize_quat_wxyz(_to_wxyz(root_rot, fmt))

    order = joint_order
    if order == "auto":
        order = _infer_joint_order_auto(dof_pos)
    joint_pos_isaac = _convert_joint_order_to_isaac(dof_pos, order)

    body_pos = _resample_linear(root_pos, source_fps, target_fps)
    body_quat_wxyz = _resample_quat_nlerp_wxyz(body_quat_wxyz, source_fps, target_fps)
    joint_pos_isaac = _resample_linear(joint_pos_isaac, source_fps, target_fps)
    joint_vel_isaac = _finite_diff(joint_pos_isaac, 1.0 / target_fps)
    smpl_joints, smpl_pose = _extract_optional_smpl_pkl(data, source_fps, target_fps)

    motion = MotionSequence(
        name=path.stem,
        fps=target_fps,
        joint_pos_isaac=joint_pos_isaac.astype(np.float64, copy=False),
        joint_vel_isaac=joint_vel_isaac.astype(np.float64, copy=False),
        body_pos=body_pos.astype(np.float64, copy=False),
        body_quat_wxyz=body_quat_wxyz.astype(np.float64, copy=False),
        smpl_joints=smpl_joints.astype(np.float64, copy=False) if smpl_joints is not None else None,
        smpl_pose=smpl_pose.astype(np.float64, copy=False) if smpl_pose is not None else None,
        encode_mode=int(encode_mode),
        resolved_joint_order=order,
    )
    motion.ensure_lengths_match()
    return motion


def _load_motion_from_hy_npz(
    path: Path,
    target_fps: float,
    encode_mode: int,
) -> MotionSequence:
    """Load a qpos `.npz` (expects qpos[T,36]) as a MotionSequence."""
    d = np.load(path, allow_pickle=False)
    if "qpos" not in d:
        raise KeyError(f"{path}: missing key 'qpos' (expected HY rollout npz)")

    qpos = np.asarray(d["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{path}: qpos must be [T,36], got {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError(f"{path}: not enough frames ({qpos.shape[0]})")
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"{path}: qpos contains NaN/Inf")

    source_fps = float(d["fps_sim"].item()) if "fps_sim" in d else float(target_fps)
    if source_fps <= 0:
        raise ValueError(f"{path}: invalid fps_sim={source_fps}")

    root_pos = qpos[:, 0:3]
    root_quat_wxyz = _normalize_quat_wxyz(qpos[:, 3:7])
    dof_pos_mujoco = qpos[:, 7:36]

    joint_pos_isaac = _convert_joint_order_to_isaac(dof_pos_mujoco, "mujoco")
    body_pos = _resample_linear(root_pos, source_fps, target_fps)
    body_quat_wxyz = _resample_quat_nlerp_wxyz(root_quat_wxyz, source_fps, target_fps)
    joint_pos_isaac = _resample_linear(joint_pos_isaac, source_fps, target_fps)
    joint_vel_isaac = _finite_diff(joint_pos_isaac, 1.0 / target_fps)

    name = path.stem
    if "sample_id" in d:
        try:
            name = str(d["sample_id"].item())
        except Exception:
            name = path.stem

    motion = MotionSequence(
        name=name,
        fps=target_fps,
        joint_pos_isaac=joint_pos_isaac.astype(np.float64, copy=False),
        joint_vel_isaac=joint_vel_isaac.astype(np.float64, copy=False),
        body_pos=body_pos.astype(np.float64, copy=False),
        body_quat_wxyz=body_quat_wxyz.astype(np.float64, copy=False),
        smpl_joints=None,
        smpl_pose=None,
        encode_mode=int(encode_mode),
        resolved_joint_order="mujoco",
    )
    motion.ensure_lengths_match()
    return motion


def _load_motion_from_dir(
    path: Path,
    motion_fps: float,
    motion_quat_order: str,
    joint_order: str,
    target_fps: float,
    encode_mode: int,
) -> MotionSequence:
    req = [path / "joint_pos.csv", path / "body_pos.csv", path / "body_quat.csv"]
    for p in req:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    joint_pos = _load_csv_matrix(path / "joint_pos.csv")[:, :29]
    joint_vel_path = path / "joint_vel.csv"
    joint_vel = _load_csv_matrix(joint_vel_path)[:, :29] if joint_vel_path.exists() else None
    body_pos = _load_csv_matrix(path / "body_pos.csv")[:, :3]
    quat_raw = _load_csv_matrix(path / "body_quat.csv")[:, :4]

    frames = min(joint_pos.shape[0], body_pos.shape[0], quat_raw.shape[0])
    if joint_vel is not None:
        frames = min(frames, joint_vel.shape[0])
    if frames < 2:
        raise ValueError(f"{path}: not enough frames ({frames})")

    joint_pos = joint_pos[:frames]
    body_pos = body_pos[:frames]
    quat_raw = quat_raw[:frames]
    if joint_vel is not None:
        joint_vel = joint_vel[:frames]

    quat_order = motion_quat_order
    if quat_order == "auto":
        quat_order = _detect_quat_format(quat_raw)
    body_quat = _normalize_quat_wxyz(_to_wxyz(quat_raw, quat_order))

    order = joint_order
    if order == "auto":
        order = _infer_joint_order_auto(joint_pos)
    joint_pos_isaac = _convert_joint_order_to_isaac(joint_pos, order)

    if joint_vel is None:
        joint_vel_isaac = _finite_diff(joint_pos_isaac, 1.0 / motion_fps)
    else:
        joint_vel_isaac = _convert_joint_order_to_isaac(joint_vel, order)

    smpl_joints = None
    smpl_pose = None
    smpl_joint_path = path / "smpl_joint.csv"
    smpl_pose_path = path / "smpl_pose.csv"
    if smpl_joint_path.exists():
        sj = _load_csv_matrix(smpl_joint_path)
        sj = sj[:frames]
        if sj.shape[1] % 3 == 0:
            smpl_joints = sj.reshape(sj.shape[0], sj.shape[1] // 3, 3)
    if smpl_pose_path.exists():
        sp = _load_csv_matrix(smpl_pose_path)
        sp = sp[:frames]
        if sp.shape[1] % 3 == 0:
            smpl_pose = sp.reshape(sp.shape[0], sp.shape[1] // 3, 3)

    if abs(motion_fps - target_fps) > 1e-9:
        body_pos = _resample_linear(body_pos, motion_fps, target_fps)
        body_quat = _resample_quat_nlerp_wxyz(body_quat, motion_fps, target_fps)
        joint_pos_isaac = _resample_linear(joint_pos_isaac, motion_fps, target_fps)
        joint_vel_isaac = _resample_linear(joint_vel_isaac, motion_fps, target_fps)
        if smpl_joints is not None:
            smpl_joints = _resample_linear(smpl_joints, motion_fps, target_fps)
        if smpl_pose is not None:
            smpl_pose = _resample_linear(smpl_pose, motion_fps, target_fps)

    motion = MotionSequence(
        name=path.name,
        fps=target_fps,
        joint_pos_isaac=np.asarray(joint_pos_isaac, dtype=np.float64),
        joint_vel_isaac=np.asarray(joint_vel_isaac, dtype=np.float64),
        body_pos=np.asarray(body_pos, dtype=np.float64),
        body_quat_wxyz=np.asarray(body_quat, dtype=np.float64),
        smpl_joints=np.asarray(smpl_joints, dtype=np.float64) if smpl_joints is not None else None,
        smpl_pose=np.asarray(smpl_pose, dtype=np.float64) if smpl_pose is not None else None,
        encode_mode=int(encode_mode),
        resolved_joint_order=order,
    )
    motion.ensure_lengths_match()
    return motion


def _load_motion_sequence(
    input_path: Path,
    input_type: str,
    root_rot_format: str,
    motion_quat_order: str,
    input_joint_order: str,
    motion_fps: float,
    target_fps: float,
    encode_mode: int,
) -> MotionSequence:
    kind = input_type
    if kind == "auto":
        if input_path.is_file() and input_path.suffix.lower() == ".pkl":
            kind = "pkl"
        elif input_path.is_file() and input_path.suffix.lower() == ".npz":
            kind = "hy-npz"
        elif input_path.is_dir() and (input_path / "joint_pos.csv").exists():
            kind = "motion-data"
        else:
            raise ValueError(f"Cannot infer input type for {input_path}")
    if kind == "pkl":
        return _load_motion_from_pkl(
            path=input_path,
            root_rot_format=root_rot_format,
            joint_order=input_joint_order,
            target_fps=target_fps,
            encode_mode=encode_mode,
        )
    if kind == "motion-data":
        return _load_motion_from_dir(
            path=input_path,
            motion_fps=motion_fps,
            motion_quat_order=motion_quat_order,
            joint_order=input_joint_order,
            target_fps=target_fps,
            encode_mode=encode_mode,
        )
    if kind == "hy-npz":
        return _load_motion_from_hy_npz(
            path=input_path,
            target_fps=target_fps,
            encode_mode=encode_mode,
        )
    raise ValueError(f"Unsupported input-type: {kind}")


def _parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z format, got: {text}")
    v = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def _parse_allowed_pred_tokens(text: str) -> np.ndarray:
    vals = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
    if len(vals) != 11:
        raise ValueError(f"--planner-allowed-pred-tokens must have 11 ints, got {len(vals)}")
    return np.asarray(vals, dtype=np.int64)


def _build_observation_specs(
    obs_config_path: Path,
    decoder_input_dim: int,
    encoder_input_dim: int,
) -> Tuple[List[ObservationSpec], int, List[ObservationSpec], Dict[int, Sequence[str]]]:
    cfg = yaml.safe_load(obs_config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid observation config: {obs_config_path}")

    enc = cfg.get("encoder", {}) or {}
    token_dim = int(enc.get("dimension", 64))

    decoder_specs: List[ObservationSpec] = []
    offset = 0
    for item in cfg.get("observations", []):
        if not item.get("enabled", True):
            continue
        name = str(item["name"])
        dim = token_dim if name == "token_state" else OBS_DIMS.get(name)
        if dim is None:
            raise KeyError(f"Unsupported observation in decoder config: {name}")
        decoder_specs.append(ObservationSpec(name=name, dim=int(dim), offset=offset))
        offset += int(dim)
    if offset != decoder_input_dim:
        raise ValueError(f"Decoder observation dim mismatch: config={offset}, model={decoder_input_dim}")

    encoder_specs: List[ObservationSpec] = []
    offset = 0
    for item in enc.get("encoder_observations", []):
        if not item.get("enabled", True):
            continue
        name = str(item["name"])
        dim = OBS_DIMS.get(name)
        if dim is None:
            raise KeyError(f"Unsupported observation in encoder config: {name}")
        encoder_specs.append(ObservationSpec(name=name, dim=int(dim), offset=offset))
        offset += int(dim)
    if offset != encoder_input_dim:
        raise ValueError(f"Encoder observation dim mismatch: config={offset}, model={encoder_input_dim}")

    encoder_modes: Dict[int, Sequence[str]] = {}
    for mode_cfg in enc.get("encoder_modes", []):
        mode_id = int(mode_cfg["mode_id"])
        required = [str(x) for x in mode_cfg.get("required_observations", [])]
        encoder_modes[mode_id] = required

    return decoder_specs, token_dim, encoder_specs, encoder_modes


class PlannerRunner:
    """Single-process ONNX planner wrapper (context update + 30Hz->50Hz conversion)."""

    def __init__(
        self,
        planner_session: ort.InferenceSession,
        look_ahead_steps: int,
        allowed_pred_tokens: np.ndarray,
    ) -> None:
        self.session = planner_session
        self.look_ahead_steps = int(look_ahead_steps)
        self.allowed_pred_tokens = np.asarray(allowed_pred_tokens, dtype=np.int64).reshape(1, 11)
        self.context_mujoco_qpos = np.zeros((1, 4, 36), dtype=np.float32)
        self.initialized = False

    def initialize_context(self, current_qpos_mujoco: np.ndarray) -> None:
        q = np.asarray(current_qpos_mujoco, dtype=np.float64).reshape(36)
        for i in range(4):
            self.context_mujoco_qpos[0, i, :] = q.astype(np.float32)
        self.initialized = True

    def _update_context_from_motion(self, gen_frame: int, motion: MotionSequence) -> None:
        if motion.timesteps <= 0:
            raise ValueError("Motion has no frames")
        gen_time = float(gen_frame) / 50.0
        for n in range(4):
            t = gen_time + float(n) / 30.0
            f50 = t * 50.0
            f0 = int(math.floor(f50))
            f0 = min(f0, motion.timesteps - 1)
            f1 = min(f0 + 1, motion.timesteps - 1)
            w1 = float(f50 - f0)
            w0 = 1.0 - w1

            p = w0 * motion.body_pos[f0] + w1 * motion.body_pos[f1]
            q = _quat_slerp_wxyz(motion.body_quat_wxyz[f0], motion.body_quat_wxyz[f1], w1)
            self.context_mujoco_qpos[0, n, 0:3] = p.astype(np.float32)
            self.context_mujoco_qpos[0, n, 3:7] = q.astype(np.float32)

            # motion.joint_pos_isaac is isaaclab order, context expects mujoco qpos order.
            q_isaac = w0 * motion.joint_pos_isaac[f0] + w1 * motion.joint_pos_isaac[f1]
            for isaac_idx in range(29):
                mj_idx = int(MUJOCO_TO_ISAACLAB[isaac_idx])
                self.context_mujoco_qpos[0, n, 7 + mj_idx] = float(q_isaac[isaac_idx])

    def update(
        self,
        current_frame: int,
        motion: MotionSequence,
        movement: MovementState,
    ) -> Optional[MotionSequence]:
        if motion.timesteps <= 1:
            return None
        if not self.initialized:
            # Bootstrap from frame 0 motion pose.
            q0_mj = np.zeros((36,), dtype=np.float64)
            q0_mj[0:3] = motion.body_pos[0]
            q0_mj[3:7] = motion.body_quat_wxyz[0]
            q0_mj[7:] = motion.joint_pos_isaac[0, ISAACLAB_TO_MUJOCO]
            self.initialize_context(q0_mj)

        gen_frame = int(current_frame) + self.look_ahead_steps
        self._update_context_from_motion(gen_frame=gen_frame, motion=motion)

        feeds: Dict[str, np.ndarray] = {
            "context_mujoco_qpos": self.context_mujoco_qpos,
            "target_vel": np.asarray([movement.movement_speed], dtype=np.float32),
            "mode": np.asarray([movement.locomotion_mode], dtype=np.int64),
            "movement_direction": movement.movement_direction.reshape(1, 3).astype(np.float32),
            "facing_direction": movement.facing_direction.reshape(1, 3).astype(np.float32),
            "random_seed": np.asarray([movement.random_seed], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": self.allowed_pred_tokens,
            "height": np.asarray([movement.height], dtype=np.float32),
        }
        out = self.session.run(None, feeds)
        qpos30 = np.asarray(out[0], dtype=np.float32).reshape(1, 64, 36)[0]
        pred_frames = int(np.asarray(out[1]).reshape(-1)[0])
        pred_frames = max(0, min(pred_frames, qpos30.shape[0]))
        if pred_frames <= 1:
            return None
        qpos30 = qpos30[:pred_frames]
        if not np.all(np.isfinite(qpos30)):
            return None

        motion_seconds = float(pred_frames) / 30.0
        frames50 = int(math.floor(motion_seconds * 50.0))
        if frames50 <= 1:
            return None

        body_pos = np.zeros((frames50, 3), dtype=np.float64)
        body_quat = np.zeros((frames50, 4), dtype=np.float64)
        joint_pos_isaac = np.zeros((frames50, 29), dtype=np.float64)

        for f50 in range(frames50):
            t = float(f50) / 50.0
            f30 = t * 30.0
            f0 = int(math.floor(f30))
            f1 = min(f0 + 1, pred_frames - 1)
            w1 = float(f30 - f0)
            w0 = 1.0 - w1

            body_pos[f50] = w0 * qpos30[f0, 0:3] + w1 * qpos30[f1, 0:3]
            body_quat[f50] = _quat_slerp_wxyz(qpos30[f0, 3:7], qpos30[f1, 3:7], w1)
            for isaac_idx in range(29):
                mj_idx = int(MUJOCO_TO_ISAACLAB[isaac_idx])
                joint_pos_isaac[f50, isaac_idx] = w0 * qpos30[f0, 7 + mj_idx] + w1 * qpos30[f1, 7 + mj_idx]

        joint_vel_isaac = np.zeros_like(joint_pos_isaac)
        joint_vel_isaac[:-1] = (joint_pos_isaac[1:] - joint_pos_isaac[:-1]) * 50.0
        joint_vel_isaac[-1] = joint_vel_isaac[-2]

        return MotionSequence(
            name="planner_motion",
            fps=50.0,
            joint_pos_isaac=joint_pos_isaac,
            joint_vel_isaac=joint_vel_isaac,
            body_pos=body_pos,
            body_quat_wxyz=_normalize_quat_wxyz(body_quat),
            smpl_joints=None,
            smpl_pose=None,
            encode_mode=motion.encode_mode,
            resolved_joint_order="isaaclab",
        )


class SonicSingleProcessPlayer:
    def __init__(
        self,
        motion: MotionSequence,
        xml_path: Path,
        decoder_session: ort.InferenceSession,
        encoder_session: ort.InferenceSession,
        planner_runner: Optional[PlannerRunner],
        decoder_obs_specs: Sequence[ObservationSpec],
        token_dim: int,
        encoder_obs_specs: Sequence[ObservationSpec],
        encoder_modes: Dict[int, Sequence[str]],
        control_decimation: int,
        sim_dt: float,
        warmup_steps: int,
        loop_motion: bool,
        planner_enabled: bool,
        planner_drive_motion: bool,
        planner_replan_interval: float,
        movement_state: MovementState,
        post_motion_behavior: str,
        print_every_seconds: float,
        save_p0_trace: bool = False,
    ) -> None:
        self.reference_motion = motion
        self.current_motion = motion
        self.current_frame = 0
        self.finished_motion = False
        self.loop_motion = bool(loop_motion)
        self.warmup_steps = int(max(0, warmup_steps))
        self.play = False

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(sim_dt)
        from ..robot_config import g1_armatures
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(1, self.model.njnt)]
        self.model.dof_armature[6:] = g1_armatures(joint_names)
        self.model.dof_frictionloss[6:] = 0
        self.model.dof_damping[6:] = 0
        self.control_decimation = int(control_decimation)
        # Match the reference's initial pose and heading before any policy action.
        self.data.qpos[:3] = motion.body_pos[0]
        self.data.qpos[3:7] = motion.body_quat_wxyz[0]
        self.data.qpos[7:36] = motion.joint_pos_isaac[0][ISAACLAB_TO_MUJOCO]
        self.data.qvel[6:35] = motion.joint_vel_isaac[0][ISAACLAB_TO_MUJOCO]
        mujoco.mj_forward(self.model, self.data)

        # Full-body action dims.
        self.n_action = 29
        if int(self.data.ctrl.shape[0]) < self.n_action:
            raise RuntimeError(f"Model ctrl dim {self.data.ctrl.shape[0]} < 29")

        self.decoder_session = decoder_session
        self.decoder_input_name = decoder_session.get_inputs()[0].name
        self.encoder_session = encoder_session
        self.encoder_input_name = encoder_session.get_inputs()[0].name
        self.planner_runner = planner_runner

        self.decoder_obs_specs = list(decoder_obs_specs)
        self.encoder_obs_specs = list(encoder_obs_specs)
        self.encoder_modes = dict(encoder_modes)
        self.token_state = np.zeros((token_dim,), dtype=np.float32)

        self.last_action_isaac = np.zeros((self.n_action,), dtype=np.float64)
        self.target_q_mujoco = DEFAULT_ANGLES_MUJOCO.copy()
        self.kps = G1_KPS.copy()
        self.kds = G1_KDS.copy()
        self.first_command_unix = None
        self.policy_calls = 0
        self.post_motion_hold_q_mujoco: Optional[np.ndarray] = None

        self.zero_state = RobotStateEntry(
            base_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            base_ang_vel_body=np.zeros((3,), dtype=np.float64),
            body_q_isaac=np.zeros((29,), dtype=np.float64),
            body_dq_isaac=np.zeros((29,), dtype=np.float64),
            last_action_isaac=np.zeros((29,), dtype=np.float64),
        )
        self.history: deque[RobotStateEntry] = deque(maxlen=400)
        for _ in range(120):
            self._append_robot_state()

        self.reinitialize_heading = True
        self.heading_state = HeadingState(init_base_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        self.init_ref_data_root_rot = self.current_motion.body_quat_wxyz[0].copy()

        self.planner_enabled = bool(planner_enabled)
        self.planner_drive_motion = bool(planner_drive_motion)
        self.planner_replan_interval = float(max(1e-3, planner_replan_interval))
        self.last_replan_time = -1e9
        self.movement_state = movement_state
        self.post_motion_behavior = str(post_motion_behavior)
        self.save_p0_trace = bool(save_p0_trace)

        self.print_every_seconds = float(max(0.1, print_every_seconds))

        self.summary: Dict[str, Any] = {
            "token_dim": int(token_dim),
            "motion_name": self.reference_motion.name,
            "resolved_joint_order": self.reference_motion.resolved_joint_order,
            "planner_enabled": self.planner_enabled,
            "planner_drive_motion": self.planner_drive_motion,
            "post_motion_behavior": self.post_motion_behavior,
            "control_decimation": int(self.control_decimation),
            "sim_dt": float(self.model.opt.timestep),
        }

    def _append_robot_state(self) -> None:
        base_quat = _normalize_quat_wxyz(self.data.qpos[3:7].copy())
        # MuJoCo free-joint qvel[3:6] is base angular velocity in the body frame.
        base_ang_vel_body = np.asarray(self.data.qvel[3:6], dtype=np.float64)

        q_mj = np.asarray(self.data.qpos[7 : 7 + self.n_action], dtype=np.float64)
        dq_mj = np.asarray(self.data.qvel[6 : 6 + self.n_action], dtype=np.float64)

        body_q_isaac = q_mj[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_ISAACLAB
        body_dq_isaac = dq_mj[MUJOCO_TO_ISAACLAB]

        entry = RobotStateEntry(
            base_quat_wxyz=base_quat,
            base_ang_vel_body=base_ang_vel_body,
            body_q_isaac=body_q_isaac,
            body_dq_isaac=body_dq_isaac,
            last_action_isaac=self.last_action_isaac.copy(),
        )
        self.history.append(entry)

    def _sample_history(self, num_frames: int, step_size: int) -> List[RobotStateEntry]:
        out: List[RobotStateEntry] = []
        hist = list(self.history)
        n = len(hist)
        for f in range(num_frames):
            offset = (num_frames - 1 - f) * step_size
            idx = n - 1 - offset
            if idx < 0 or idx >= n:
                out.append(self.zero_state)
            else:
                out.append(hist[idx])
        return out

    def _motion_target_frame(self, frame_idx: int, step_size: int) -> int:
        f = int(self.current_frame)
        if self.play:
            f += frame_idx * step_size
        return min(max(f, 0), self.current_motion.timesteps - 1)

    def _gather_motion_joint_pos(self, num_frames: int, step_size: int, joint_idx: Optional[np.ndarray] = None) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for f in range(num_frames):
            t = self._motion_target_frame(f, step_size)
            row = self.current_motion.joint_pos_isaac[t]
            if joint_idx is not None:
                row = row[joint_idx]
            chunks.append(np.asarray(row, dtype=np.float64))
        return np.concatenate(chunks, axis=0)

    def _gather_motion_joint_vel(self, num_frames: int, step_size: int, joint_idx: Optional[np.ndarray] = None) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for f in range(num_frames):
            t = self._motion_target_frame(f, step_size)
            if self.play:
                row = self.current_motion.joint_vel_isaac[t]
                if joint_idx is not None:
                    row = row[joint_idx]
            else:
                row = np.zeros((len(joint_idx) if joint_idx is not None else 29,), dtype=np.float64)
            chunks.append(np.asarray(row, dtype=np.float64))
        return np.concatenate(chunks, axis=0)

    def _gather_motion_root_z(self, num_frames: int, step_size: int) -> np.ndarray:
        out = np.zeros((num_frames,), dtype=np.float64)
        for f in range(num_frames):
            t = self._motion_target_frame(f, step_size)
            out[f] = float(self.current_motion.body_pos[t, 2])
        return out

    def _gather_motion_anchor_orientation(self, num_frames: int, step_size: int, orientation_mode: int = 0) -> np.ndarray:
        latest = self.history[-1] if len(self.history) > 0 else self.zero_state
        base_quat = latest.base_quat_wxyz

        if self.reinitialize_heading:
            self.heading_state = HeadingState(init_base_quat_wxyz=base_quat.copy(), delta_heading=0.0)
            self.reinitialize_heading = False
            self.init_ref_data_root_rot = self.current_motion.body_quat_wxyz[self.current_frame].copy()

        if self.current_frame == 0:
            self.init_ref_data_root_rot = self.current_motion.body_quat_wxyz[0].copy()

        init_heading = _calc_heading_quat_wxyz(self.heading_state.init_base_quat_wxyz)
        data_heading_inv = _calc_heading_quat_inv_wxyz(self.init_ref_data_root_rot)
        apply_delta_heading = _quat_mul_wxyz(init_heading, data_heading_inv)

        if abs(self.heading_state.delta_heading) > 1e-9:
            dq = _quat_from_angle_axis(self.heading_state.delta_heading, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
            apply_delta_heading = _quat_mul_wxyz(dq, apply_delta_heading)

        out = np.zeros((num_frames * 6,), dtype=np.float64)
        for f in range(num_frames):
            t = self._motion_target_frame(f, step_size)
            ref_root = self.current_motion.body_quat_wxyz[t]
            new_ref_root = _quat_mul_wxyz(apply_delta_heading, ref_root)
            # Upstream SONIC v1.1 g1_deploy_onnx_ref.cpp orientation_mode=1.
            left_quat = _calc_heading_quat_wxyz(base_quat) if orientation_mode == 1 else base_quat
            base_to_ref = _quat_mul_wxyz(_quat_conjugate_wxyz(left_quat), new_ref_root)
            rot = _quat_to_rotmat_wxyz(base_to_ref)
            o = f * 6
            out[o + 0] = rot[0, 0]
            out[o + 1] = rot[0, 1]
            out[o + 2] = rot[1, 0]
            out[o + 3] = rot[1, 1]
            out[o + 4] = rot[2, 0]
            out[o + 5] = rot[2, 1]
        return out

    def _gather_history_body_q(self, num_frames: int, step_size: int) -> np.ndarray:
        hist = self._sample_history(num_frames=num_frames, step_size=step_size)
        return np.concatenate([h.body_q_isaac for h in hist], axis=0).astype(np.float64, copy=False)

    def _gather_history_body_dq(self, num_frames: int, step_size: int) -> np.ndarray:
        hist = self._sample_history(num_frames=num_frames, step_size=step_size)
        return np.concatenate([h.body_dq_isaac for h in hist], axis=0).astype(np.float64, copy=False)

    def _gather_history_last_action(self, num_frames: int, step_size: int) -> np.ndarray:
        hist = self._sample_history(num_frames=num_frames, step_size=step_size)
        return np.concatenate([h.last_action_isaac for h in hist], axis=0).astype(np.float64, copy=False)

    def _gather_history_base_ang_vel(self, num_frames: int, step_size: int) -> np.ndarray:
        hist = self._sample_history(num_frames=num_frames, step_size=step_size)
        return np.concatenate([h.base_ang_vel_body for h in hist], axis=0).astype(np.float64, copy=False)

    def _gather_history_gravity(self, num_frames: int, step_size: int) -> np.ndarray:
        hist = self._sample_history(num_frames=num_frames, step_size=step_size)
        out = []
        g = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        for h in hist:
            out.append(_quat_rotate_inverse_wxyz(h.base_quat_wxyz, g))
        return np.concatenate(out, axis=0).astype(np.float64, copy=False)

    def _gather_vr3_pos(self) -> np.ndarray:
        # No external teleop in this script. Keep zero target.
        return np.zeros((9,), dtype=np.float64)

    def _gather_vr3_orn(self) -> np.ndarray:
        # No external teleop in this script. Keep zero target.
        return np.zeros((12,), dtype=np.float64)

    def _gather_smpl_joints(self, num_frames: int, step_size: int) -> Optional[np.ndarray]:
        if self.current_motion.smpl_joints is None:
            return None
        sj = self.current_motion.smpl_joints
        if sj.shape[1] * 3 != 72:
            return None
        out = np.zeros((num_frames * 72,), dtype=np.float64)
        for f in range(num_frames):
            t = self._motion_target_frame(f, step_size)
            out[f * 72 : (f + 1) * 72] = sj[t].reshape(-1)
        return out

    def _gather_observation(self, name: str, encoder_mode: Optional[int] = None) -> Optional[np.ndarray]:
        if name == "token_state":
            return self.token_state.astype(np.float64, copy=False)
        if name == "his_base_angular_velocity_10frame_step1":
            return self._gather_history_base_ang_vel(10, 1)
        if name == "his_body_joint_positions_10frame_step1":
            return self._gather_history_body_q(10, 1)
        if name == "his_body_joint_velocities_10frame_step1":
            return self._gather_history_body_dq(10, 1)
        if name == "his_last_actions_10frame_step1":
            return self._gather_history_last_action(10, 1)
        if name == "his_gravity_dir_10frame_step1":
            return self._gather_history_gravity(10, 1)
        if name == "encoder_mode_4":
            mode = int(self.current_motion.encode_mode if encoder_mode is None else encoder_mode)
            return np.asarray([float(mode), 0.0, 0.0, 0.0], dtype=np.float64)
        if name == "motion_joint_positions_10frame_step5":
            return self._gather_motion_joint_pos(10, 5)
        if name == "motion_joint_velocities_10frame_step5":
            return self._gather_motion_joint_vel(10, 5)
        if name == "motion_root_z_position_10frame_step5":
            return self._gather_motion_root_z(10, 5)
        if name == "motion_root_z_position":
            return self._gather_motion_root_z(1, 1)
        if name == "motion_anchor_orientation":
            return self._gather_motion_anchor_orientation(1, 1)
        if name == "motion_anchor_orientation_10frame_step5":
            return self._gather_motion_anchor_orientation(10, 5)
        if name == "motion_anchor_orientation_heading":
            return self._gather_motion_anchor_orientation(1, 1, orientation_mode=1)
        if name == "motion_anchor_orientation_heading_10frame_step5":
            return self._gather_motion_anchor_orientation(10, 5, orientation_mode=1)
        if name == "motion_joint_positions_lowerbody_10frame_step5":
            return self._gather_motion_joint_pos(10, 5, LOWER_BODY_ISAACLAB_INDEX)
        if name == "motion_joint_velocities_lowerbody_10frame_step5":
            return self._gather_motion_joint_vel(10, 5, LOWER_BODY_ISAACLAB_INDEX)
        if name == "vr_3point_local_target":
            return self._gather_vr3_pos()
        if name == "vr_3point_local_orn_target":
            return self._gather_vr3_orn()
        if name == "smpl_joints_10frame_step1":
            return self._gather_smpl_joints(10, 1)
        if name == "smpl_anchor_orientation_10frame_step1":
            return self._gather_motion_anchor_orientation(10, 1)
        if name == "smpl_anchor_orientation_heading_10frame_step1":
            return self._gather_motion_anchor_orientation(10, 1, orientation_mode=1)
        if name == "motion_joint_positions_wrists_10frame_step1":
            return self._gather_motion_joint_pos(10, 1, WRIST_ISAACLAB_INDEX)
        return None

    def _run_encoder(self) -> None:
        intended_mode = int(self.current_motion.encode_mode)
        modes_to_try: List[int] = []
        if intended_mode >= 0:
            modes_to_try.append(intended_mode)
        for mode_id in self.encoder_modes.keys():
            if mode_id not in modes_to_try:
                modes_to_try.append(int(mode_id))
        if not modes_to_try:
            modes_to_try.append(-1)

        last_err: Optional[str] = None
        for mode in modes_to_try:
            obs = np.zeros((self.encoder_session.get_inputs()[0].shape[1],), dtype=np.float64)
            required = self.encoder_modes.get(mode, None)
            ok = True
            for spec in self.encoder_obs_specs:
                if required is not None and spec.name not in required:
                    continue
                v = self._gather_observation(spec.name, encoder_mode=mode)
                if v is None:
                    ok = False
                    last_err = f"Missing encoder observation: {spec.name} for mode={mode}"
                    break
                v = np.asarray(v, dtype=np.float64).reshape(-1)
                if v.shape[0] != spec.dim:
                    ok = False
                    last_err = (
                        f"Encoder observation dim mismatch for {spec.name}: "
                        f"got={v.shape[0]} expected={spec.dim}"
                    )
                    break
                obs[spec.offset : spec.offset + spec.dim] = v
            if not ok:
                continue
            out = self.encoder_session.run(None, {self.encoder_input_name: obs[None, :].astype(np.float32)})[0]
            token = np.asarray(out, dtype=np.float32).reshape(-1)
            if token.shape[0] != self.token_state.shape[0]:
                raise RuntimeError(f"Encoder token dim mismatch: got={token.shape[0]} expected={self.token_state.shape[0]}")
            self.token_state = token
            self.current_motion.encode_mode = mode
            return

        raise RuntimeError(last_err or "Failed to gather encoder observations")

    def _build_decoder_obs(self) -> np.ndarray:
        dim = int(self.decoder_session.get_inputs()[0].shape[1])
        obs = np.zeros((dim,), dtype=np.float64)
        for spec in self.decoder_obs_specs:
            v = self._gather_observation(spec.name, encoder_mode=None)
            if v is None:
                raise RuntimeError(f"Missing decoder observation: {spec.name}")
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            if v.shape[0] != spec.dim:
                raise RuntimeError(
                    f"Decoder observation dim mismatch for {spec.name}: got={v.shape[0]} expected={spec.dim}"
                )
            obs[spec.offset : spec.offset + spec.dim] = v
        return obs[None, :].astype(np.float32)

    def _decoder_step(self) -> None:
        self._run_encoder()
        dec_obs = self._build_decoder_obs()
        out = self.decoder_session.run(None, {self.decoder_input_name: dec_obs})[0]
        action_isaac = np.asarray(out, dtype=np.float64).reshape(-1)
        if action_isaac.shape[0] != self.n_action:
            raise RuntimeError(f"Decoder output dim mismatch: got={action_isaac.shape[0]} expected={self.n_action}")
        self.last_action_isaac = action_isaac.copy()
        self.target_q_mujoco = DEFAULT_ANGLES_MUJOCO + action_isaac[ISAACLAB_TO_MUJOCO] * G1_ACTION_SCALE
        self.policy_calls += 1
        if self.first_command_unix is None:
            self.first_command_unix = time.time()

    def _apply_pd(self) -> None:
        q = np.asarray(self.data.qpos[7 : 7 + self.n_action], dtype=np.float64)
        dq = np.asarray(self.data.qvel[6 : 6 + self.n_action], dtype=np.float64)
        tau = (self.target_q_mujoco - q) * self.kps + (0.0 - dq) * self.kds
        self.data.ctrl[: self.n_action] = tau
        if int(self.data.ctrl.shape[0]) > self.n_action:
            self.data.ctrl[self.n_action :] = 0.0

    def _maybe_replan(self, control_time_s: float) -> None:
        if not self.planner_enabled or self.planner_runner is None:
            return
        if (control_time_s - self.last_replan_time) < self.planner_replan_interval:
            return
        repl_motion = self.planner_runner.update(
            current_frame=self.current_frame,
            motion=self.current_motion,
            movement=self.movement_state,
        )
        self.last_replan_time = control_time_s
        if repl_motion is None:
            return
        if self.planner_drive_motion:
            self.current_motion = repl_motion
            self.current_frame = 0
            self.reinitialize_heading = True
            self.summary["planner_last_pred_frames"] = int(repl_motion.timesteps)
        else:
            # Keep planner result only for diagnostics in summary.
            self.summary["planner_last_pred_frames"] = int(repl_motion.timesteps)

    def _advance_frame(self) -> None:
        if not self.play:
            return
        self.current_frame += 1
        if self.current_frame < self.current_motion.timesteps:
            return
        if self.loop_motion:
            self.current_frame = 0
            self.reinitialize_heading = True
            self.post_motion_hold_q_mujoco = None
            return
        self.current_frame = 0 if self.post_motion_behavior == "deploy" else self.current_motion.timesteps - 1
        if self.post_motion_behavior == "deploy":
            self.reinitialize_heading = True
        self.play = False
        self.finished_motion = True

    def _handle_post_motion(self) -> bool:
        """Handle behavior after motion clip is finished.

        Returns:
            True when simulation loop should stop immediately.
        """
        mode = self.post_motion_behavior
        if mode in ("decode", "deploy"):
            return False
        if mode == "stop":
            self.play = False
            self.last_action_isaac[:] = 0.0
            self.target_q_mujoco = DEFAULT_ANGLES_MUJOCO.copy()
            return True
        # Default safety behavior: hold current pose to avoid post-clip jump/fall.
        self.play = False
        self.last_action_isaac[:] = 0.0
        if self.post_motion_hold_q_mujoco is None:
            self.post_motion_hold_q_mujoco = np.asarray(self.data.qpos[7 : 7 + self.n_action], dtype=np.float64).copy()
        self.target_q_mujoco = self.post_motion_hold_q_mujoco.copy()
        return False

    def run(
        self,
        out_video: Optional[Path],
        video_fps: float,
        video_width: int,
        video_height: int,
        camera_distance: float,
        camera_azimuth: float,
        camera_elevation: float,
        max_seconds: float,
        stop_when_motion_ends: bool,
    ) -> Dict[str, Any]:
        # Run length in sim ticks.
        if max_seconds > 0:
            total_sim_steps = max(1, int(round(max_seconds / self.model.opt.timestep)))
        else:
            ref_control_steps = max(1, self.reference_motion.timesteps + self.warmup_steps)
            total_sim_steps = ref_control_steps * self.control_decimation

        writer = None
        renderer = None
        camera = None
        gl_ctx = None
        render_w = int(video_width)
        render_h = int(video_height)
        video_stride = max(1, int(round(1.0 / (float(video_fps) * self.model.opt.timestep))))
        if out_video is not None:
            out_video.parent.mkdir(parents=True, exist_ok=True)
            off_w = int(self.model.vis.global_.offwidth)
            off_h = int(self.model.vis.global_.offheight)
            if render_w > off_w or render_h > off_h:
                render_w = min(render_w, off_w)
                render_h = min(render_h, off_h)
                print(
                    f"[WARN] video size {video_width}x{video_height} exceeds offscreen buffer "
                    f"{off_w}x{off_h}; using {render_w}x{render_h}"
                )
            gl_ctx = mujoco.GLContext(render_w, render_h)
            gl_ctx.make_current()
            renderer = mujoco.Renderer(self.model, width=render_w, height=render_h)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.distance = float(camera_distance)
            camera.azimuth = float(camera_azimuth)
            camera.elevation = float(camera_elevation)
            writer = imageio.get_writer(str(out_video), fps=float(video_fps))

        print(
            f"[INFO] single-process SONIC run started: sim_steps={total_sim_steps}, "
            f"sim_dt={self.model.opt.timestep:.4f}, control_decimation={self.control_decimation}"
        )
        start_t = time.time()
        next_log_t = start_t + self.print_every_seconds

        base_z_hist: List[float] = []
        upper_body_motion: List[np.ndarray] = []
        joint_motion: List[np.ndarray] = []
        action_norm_hist: List[float] = []
        action_hist: List[np.ndarray] = []
        target_q_hist: List[np.ndarray] = []
        ctrl_hist: List[np.ndarray] = []
        motion_root_z_hist: List[float] = []
        rollout_qpos_hist: List[np.ndarray] = []
        rollout_qvel_hist: List[np.ndarray] = []
        rendered_frames = 0
        control_steps = 0
        stop_requested = False

        for sim_step in range(total_sim_steps):
            if sim_step % self.control_decimation == 0:
                control_steps += 1
                control_time_s = (control_steps - 1) / 50.0
                self.play = control_steps > self.warmup_steps and not self.finished_motion

                # Log state at control tick (50Hz) in HY rollout layout:
                # qpos = root_pos(3) + root_quat(wxyz,4) + dof_pos(29) in MuJoCo order
                # qvel = root_vel(3) + root_angvel(3) + dof_vel(29) in MuJoCo order
                if int(self.model.nq) >= 36:
                    rollout_qpos_hist.append(np.asarray(self.data.qpos[:36], dtype=np.float64).copy())
                if int(self.model.nv) >= 35:
                    rollout_qvel_hist.append(np.asarray(self.data.qvel[:35], dtype=np.float64).copy())

                self._append_robot_state()
                if self.finished_motion and self.post_motion_behavior not in ("decode", "deploy"):
                    stop_requested = self._handle_post_motion()
                else:
                    self._maybe_replan(control_time_s)
                    self._decoder_step()
                action_norm_hist.append(float(np.linalg.norm(self.last_action_isaac)))
                action_hist.append(self.last_action_isaac.copy())
                target_q_hist.append(self.target_q_mujoco.copy())
                if self.save_p0_trace:
                    motion_root_z_hist.append(float(self._gather_motion_root_z(1, 1)[0]))
                self._advance_frame()
                if stop_requested:
                    break

            self._apply_pd()
            ctrl_hist.append(np.asarray(self.data.ctrl[:29], dtype=np.float64).copy())
            mujoco.mj_step(self.model, self.data)

            base_z_hist.append(float(self.data.qpos[2]))
            upper_body_motion.append(np.asarray(self.data.qpos[7 + 12 : 7 + 29], dtype=np.float64).copy())
            joint_motion.append(np.asarray(self.data.qpos[7 : 7 + 29], dtype=np.float64).copy())

            if writer is not None and renderer is not None and camera is not None and (sim_step % video_stride == 0):
                camera.lookat[:] = self.data.qpos[0:3]
                renderer.update_scene(self.data, camera=camera)
                writer.append_data(renderer.render())
                rendered_frames += 1

            now = time.time()
            if now >= next_log_t:
                progress = (sim_step + 1) / float(total_sim_steps)
                print(
                    f"[INFO] progress={progress*100:.1f}% step={sim_step+1}/{total_sim_steps} "
                    f"frame={self.current_frame} z={self.data.qpos[2]:.3f} play={self.play}"
                )
                next_log_t = now + self.print_every_seconds

            if stop_when_motion_ends and self.finished_motion and (sim_step + 1) % self.control_decimation == 0:
                break

        # Keep the state after the last full control interval, too. The preceding
        # samples are pre-command states, so this makes the trajectory N+1 for N actions.
        if len(ctrl_hist) % self.control_decimation == 0 and not stop_requested:
            rollout_qpos_hist.append(np.asarray(self.data.qpos[:36], dtype=np.float64).copy())
            rollout_qvel_hist.append(np.asarray(self.data.qvel[:35], dtype=np.float64).copy())
        elapsed = time.time() - start_t
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if gl_ctx is not None:
            try:
                gl_ctx.free()
            except Exception:
                pass

        upper_motion_arr = np.asarray(upper_body_motion, dtype=np.float64) if upper_body_motion else np.zeros((1, 17))
        joint_motion_arr = np.asarray(joint_motion, dtype=np.float64) if joint_motion else np.zeros((1, 29))
        action_arr = np.asarray(action_hist, dtype=np.float64) if action_hist else np.zeros((1, 29))
        target_q_arr = np.asarray(target_q_hist, dtype=np.float64) if target_q_hist else np.zeros((1, 29))
        ctrl_arr = np.asarray(ctrl_hist, dtype=np.float64) if ctrl_hist else np.zeros((1, 29))

        summary = {
            "elapsed_seconds": float(elapsed),
            "simulated_seconds": float((sim_step + 1) * self.model.opt.timestep),
            "sim_steps": int(sim_step + 1),
            "control_steps": int(control_steps),
            "video_frames": int(rendered_frames),
            "video_width": int(render_w) if out_video is not None else 0,
            "video_height": int(render_h) if out_video is not None else 0,
            "base_z_min": float(np.min(base_z_hist)) if base_z_hist else 0.0,
            "base_z_max": float(np.max(base_z_hist)) if base_z_hist else 0.0,
            "base_z_mean": float(np.mean(base_z_hist)) if base_z_hist else 0.0,
            "upper_body_q_std_mean": float(np.mean(np.std(upper_motion_arr, axis=0))),
            "joint_q_std_upper_mean": float(np.mean(np.std(joint_motion_arr[:, 12:], axis=0))),
            "joint_q_std_lower_mean": float(np.mean(np.std(joint_motion_arr[:, :12], axis=0))),
            "action_std_mean": float(np.mean(np.std(action_arr, axis=0))),
            "action_std_upper_mean": float(np.mean(np.std(action_arr[:, 12:], axis=0))),
            "action_std_lower_mean": float(np.mean(np.std(action_arr[:, :12], axis=0))),
            "target_q_std_upper_mean": float(np.mean(np.std(target_q_arr[:, 12:], axis=0))),
            "target_q_meanabs_upper_mean": float(np.mean(np.abs(np.mean(target_q_arr[:, 12:], axis=0)))),
            "target_q_meanabs_lower_mean": float(np.mean(np.abs(np.mean(target_q_arr[:, :12], axis=0)))),
            "ctrl_std_upper_mean": float(np.mean(np.std(ctrl_arr[:, 12:], axis=0))),
            "action_norm_mean": float(np.mean(action_norm_hist)) if action_norm_hist else 0.0,
            "action_norm_max": float(np.max(action_norm_hist)) if action_norm_hist else 0.0,
            "final_motion_name": self.current_motion.name,
            "final_frame": int(self.current_frame),
            "motion_finished": bool(self.finished_motion),
            "play_flag": bool(self.play),
            "encode_mode_intended": int(self.reference_motion.encode_mode),
            "encode_mode_final": int(self.current_motion.encode_mode),
            "has_smpl_joints": bool(self.current_motion.smpl_joints is not None),
        }
        summary.update(self.summary)
        # Expose rollout arrays for optional export from main().
        self.rollout_qpos = (
            np.asarray(rollout_qpos_hist, dtype=np.float32) if rollout_qpos_hist else np.zeros((1, 36), dtype=np.float32)
        )
        self.rollout_qvel = (
            np.asarray(rollout_qvel_hist, dtype=np.float32) if rollout_qvel_hist else np.zeros((1, 35), dtype=np.float32)
        )
        if self.save_p0_trace:
            self.p0_action_isaac_hist = np.asarray(action_hist, dtype=np.float32) if action_hist else np.zeros((0, 29), dtype=np.float32)
            self.p0_target_q_mujoco_hist = (
                np.asarray(target_q_hist, dtype=np.float32) if target_q_hist else np.zeros((0, 29), dtype=np.float32)
            )
            self.p0_ctrl_hist = np.asarray(ctrl_hist, dtype=np.float32) if ctrl_hist else np.zeros((0, 29), dtype=np.float32)
            self.p0_base_z_hist = np.asarray(base_z_hist, dtype=np.float32) if base_z_hist else np.zeros((0,), dtype=np.float32)
            self.p0_motion_root_z_hist = (
                np.asarray(motion_root_z_hist, dtype=np.float32) if motion_root_z_hist else np.zeros((0,), dtype=np.float32)
            )
        return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-process SONIC MuJoCo play (encoder + planner + decoder)")
    p.add_argument("--input", required=True, help="Input .pkl / HY rollout .npz / motion-data clip dir")
    p.add_argument("--input-type", choices=["auto", "pkl", "hy-npz", "motion-data"], default="auto")
    p.add_argument("--repo-root", default=str(DEFAULT_REPO), help="Repo root")
    p.add_argument("--xml-path", default=str(DEFAULT_XML), help="MuJoCo XML path")
    p.add_argument("--obs-config", default=str(DEFAULT_OBS_CONFIG), help="observation_config.yaml")
    p.add_argument("--encoder-onnx", default=str(DEFAULT_ENCODER_ONNX), help="Encoder ONNX")
    p.add_argument("--decoder-onnx", default=str(DEFAULT_DECODER_ONNX), help="Decoder ONNX")
    p.add_argument("--planner-onnx", default=str(DEFAULT_PLANNER_ONNX), help="Planner ONNX")

    p.add_argument("--root-rot-format", choices=["auto", "xyzw", "wxyz"], default="xyzw")
    p.add_argument("--motion-quat-order", choices=["auto", "xyzw", "wxyz"], default="wxyz")
    p.add_argument("--input-joint-order", choices=["auto", "mujoco", "isaaclab"], default="auto")
    p.add_argument("--motion-fps", type=float, default=50.0, help="Input FPS when --input-type motion-data")
    p.add_argument("--target-fps", type=float, default=50.0, help="Resample input motion to target fps")
    p.add_argument("--encoder-mode", type=int, default=0, help="Initial encode_mode (0=g1, 2=smpl)")

    p.add_argument("--sim-dt", type=float, default=0.005, help="MuJoCo dt")
    p.add_argument("--control-decimation", type=int, default=4, help="Sim steps per control tick")
    p.add_argument("--control-hz", type=float, default=50.0, help="Expected control rate (for sanity check)")
    p.add_argument("--warmup-seconds", type=float, default=0.5, help="Warmup (play=false) before advancing motion")
    p.add_argument("--loop-motion", action="store_true", help="Loop motion clip")
    p.add_argument(
        "--stop-when-motion-ends",
        dest="stop_when_motion_ends",
        action="store_true",
        default=True,
        help="Stop run after one motion pass (default: enabled).",
    )
    p.add_argument(
        "--no-stop-when-motion-ends",
        dest="stop_when_motion_ends",
        action="store_false",
        help="Keep running after motion end (may be unstable for some clips).",
    )

    p.add_argument("--disable-planner", action="store_true", help="Disable planner ONNX")
    p.add_argument("--planner-drive-motion", action="store_true", help="Use planner output as current motion")
    p.add_argument("--planner-replan-interval", type=float, default=1.0, help="Planner replanning interval seconds")
    p.add_argument("--planner-look-ahead-steps", type=int, default=2, help="Planner context look-ahead steps")
    p.add_argument("--planner-mode", type=int, default=0, help="Planner locomotion mode")
    p.add_argument("--planner-target-vel", type=float, default=-1.0, help="Planner target velocity")
    p.add_argument("--planner-height", type=float, default=-1.0, help="Planner target height")
    p.add_argument("--planner-movement-dir", default="0,0,0", help="Planner movement direction x,y,z")
    p.add_argument("--planner-facing-dir", default="1,0,0", help="Planner facing direction x,y,z")
    p.add_argument("--planner-random-seed", type=int, default=1234, help="Planner random seed")
    p.add_argument(
        "--planner-allowed-pred-tokens",
        default="1,1,1,1,1,1,0,0,0,0,0",
        help="Planner allowed_pred_num_tokens (11 ints)",
    )

    p.add_argument("--max-seconds", type=float, default=0.0, help="Max simulation seconds (<=0 means clip-driven)")
    p.add_argument(
        "--post-motion-behavior",
        choices=["stand", "decode", "deploy", "stop"],
        default="stand",
        help="Behavior after clip ends: stand (safe default), decode (last frame), deploy (pause frame zero and reset heading), stop (end run).",
    )
    p.add_argument("--output-video", default="", help="Output mp4 path")
    p.add_argument("--video-fps", type=float, default=20.0)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--camera-distance", type=float, default=4.0)
    p.add_argument("--camera-azimuth", type=float, default=90.0)
    p.add_argument("--camera-elevation", type=float, default=-20.0)
    p.add_argument("--print-every-seconds", type=float, default=2.0)
    p.add_argument("--summary-json", default="", help="Path to write run summary json")
    p.add_argument("--save-p0-trace", action="store_true", help="Keep per-step action/target/torque traces on the player.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    input_path = Path(args.input).resolve()

    def _resolve_under_repo(path_text: str) -> Path:
        p = Path(path_text).expanduser()
        if not p.is_absolute():
            p = repo_root / p
        return p.resolve()

    xml_path = _resolve_under_repo(str(args.xml_path))
    obs_config = _resolve_under_repo(str(args.obs_config))
    encoder_onnx = _resolve_under_repo(str(args.encoder_onnx))
    decoder_onnx = _resolve_under_repo(str(args.decoder_onnx))
    planner_onnx = _resolve_under_repo(str(args.planner_onnx))
    out_video = Path(args.output_video).resolve() if args.output_video else None

    for p in [repo_root, input_path, xml_path, obs_config, encoder_onnx, decoder_onnx]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")
    if not args.disable_planner and not planner_onnx.exists():
        raise FileNotFoundError(f"Planner ONNX not found: {planner_onnx}")

    effective_control_hz = 1.0 / (float(args.sim_dt) * int(args.control_decimation))
    if abs(effective_control_hz - float(args.control_hz)) > 1e-6:
        print(
            f"[WARN] control-hz mismatch: expected={float(args.control_hz):.6f}, "
            f"effective(sim_dt*decimation)={effective_control_hz:.6f}. Using effective value."
        )

    target_fps = float(args.target_fps)
    if abs(target_fps - effective_control_hz) > 1e-6:
        print(
            f"[WARN] target-fps={target_fps:.6f} differs from effective control_hz={effective_control_hz:.6f}. "
            "Using control_hz for motion resampling."
        )
        target_fps = effective_control_hz

    motion = _load_motion_sequence(
        input_path=input_path,
        input_type=str(args.input_type),
        root_rot_format=str(args.root_rot_format),
        motion_quat_order=str(args.motion_quat_order),
        input_joint_order=str(args.input_joint_order),
        motion_fps=float(args.motion_fps),
        target_fps=target_fps,
        encode_mode=int(args.encoder_mode),
    )
    print(
        f"[INFO] motion loaded: name={motion.name}, frames={motion.timesteps}, fps={motion.fps:.3f}, "
        f"joint_order={motion.resolved_joint_order}, smpl_joints={'yes' if motion.smpl_joints is not None else 'no'}"
    )

    decoder_sess = _make_ort_session(decoder_onnx)
    encoder_sess = _make_ort_session(encoder_onnx)
    dec_in_shape = decoder_sess.get_inputs()[0].shape
    enc_in_shape = encoder_sess.get_inputs()[0].shape
    dec_in_dim = int(dec_in_shape[1])
    enc_in_dim = int(enc_in_shape[1])
    dec_out_dim = int(decoder_sess.get_outputs()[0].shape[1])
    enc_out_dim = int(encoder_sess.get_outputs()[0].shape[1])
    if dec_out_dim != 29:
        raise RuntimeError(f"Decoder output dim must be 29, got {dec_out_dim}")
    print(f"[INFO] decoder_input_dim={dec_in_dim}, encoder_input_dim={enc_in_dim}, token_dim={enc_out_dim}")

    decoder_specs, token_dim_cfg, encoder_specs, encoder_modes = _build_observation_specs(
        obs_config_path=obs_config,
        decoder_input_dim=dec_in_dim,
        encoder_input_dim=enc_in_dim,
    )
    if token_dim_cfg != enc_out_dim:
        raise RuntimeError(f"Token dim mismatch: config={token_dim_cfg} encoder_output={enc_out_dim}")

    planner_runner: Optional[PlannerRunner] = None
    planner_enabled = not bool(args.disable_planner)
    if planner_enabled:
        planner_sess = _make_ort_session(planner_onnx)
        planner_runner = PlannerRunner(
            planner_session=planner_sess,
            look_ahead_steps=int(args.planner_look_ahead_steps),
            allowed_pred_tokens=_parse_allowed_pred_tokens(str(args.planner_allowed_pred_tokens)),
        )

    movement = MovementState(
        locomotion_mode=int(args.planner_mode),
        movement_direction=_parse_vec3(str(args.planner_movement_dir)),
        facing_direction=_parse_vec3(str(args.planner_facing_dir)),
        movement_speed=float(args.planner_target_vel),
        height=float(args.planner_height),
        random_seed=int(args.planner_random_seed),
    )

    warmup_steps = int(round(float(args.warmup_seconds) * effective_control_hz))
    player = SonicSingleProcessPlayer(
        motion=motion,
        xml_path=xml_path,
        decoder_session=decoder_sess,
        encoder_session=encoder_sess,
        planner_runner=planner_runner,
        decoder_obs_specs=decoder_specs,
        token_dim=token_dim_cfg,
        encoder_obs_specs=encoder_specs,
        encoder_modes=encoder_modes,
        control_decimation=int(args.control_decimation),
        sim_dt=float(args.sim_dt),
        warmup_steps=warmup_steps,
        loop_motion=bool(args.loop_motion),
        planner_enabled=planner_enabled,
        planner_drive_motion=bool(args.planner_drive_motion),
        planner_replan_interval=float(args.planner_replan_interval),
        movement_state=movement,
        post_motion_behavior=str(args.post_motion_behavior),
        print_every_seconds=float(args.print_every_seconds),
        save_p0_trace=bool(args.save_p0_trace),
    )

    summary = player.run(
        out_video=out_video,
        video_fps=float(args.video_fps),
        video_width=int(args.video_width),
        video_height=int(args.video_height),
        camera_distance=float(args.camera_distance),
        camera_azimuth=float(args.camera_azimuth),
        camera_elevation=float(args.camera_elevation),
        max_seconds=float(args.max_seconds),
        stop_when_motion_ends=bool(args.stop_when_motion_ends),
    )

    summary.update(
        {
            "input": str(input_path),
            "input_type": str(args.input_type),
            "output_video": str(out_video) if out_video is not None else "",
            "single_process": True,
            "encoder_onnx": str(encoder_onnx),
            "decoder_onnx": str(decoder_onnx),
            "planner_onnx": str(planner_onnx) if planner_enabled else "",
        }
    )
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_json:
        summary_path = Path(args.summary_json).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] summary written: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
