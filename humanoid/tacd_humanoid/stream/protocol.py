"""SONIC ZMQ stream Protocol v1 (G1 whole-body joint targets), sender side.

Wire format (gear_sonic_deploy zmq_packed_message_subscriber.hpp / zmq_endpoint_interface.hpp and
docs/source/tutorials/zmq.md, upstream commit 7f151314):

    one ZMQ frame = topic bytes ("pose") + 1280-byte JSON header (NUL padded) + concatenated
    little-endian C-contiguous field buffers, in header order.
    header = {"v": 1, "endian": "le", "count": 1,
              "fields": [{"name": ..., "dtype": "f32"|"f64"|"i32"|"i64"|"bool", "shape": [...]}, ...]}

Protocol v1 fields (encode mode 0, the g1 joint-based encoder of sonic_v1_1):
    joint_pos  [N,29] f32  IsaacLab joint order, radians
    joint_vel  [N,29] f32  IsaacLab joint order, rad/s
    body_quat  [N,4]  f32  root (pelvis) world orientation, w x y z
    frame_index [N]   i64  global, monotonically increasing 50 Hz frame numbers
optional:
    catch_up   [1]   bool  (default true on the robot side)
    timestamp_monotonic [1] f64  CLOCK_MONOTONIC seconds at send time (deploy only uses it for
                                 streaming-latency statistics when the publisher is on localhost)

The deploy's StreamedMotionMerger aligns successive messages on frame_index and advances its own
cursor by one frame per 50 Hz control tick, so the publisher sends overlapping windows of future
frames (a "lookahead") rather than one frame per message.
"""
import json

import numpy as np

HEADER_SIZE = 1280
PROTOCOL_VERSION = 1
CONTROL_FPS = 50
NUM_JOINTS = 29
_DTYPES = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64", np.dtype(np.int32): "i32",
           np.dtype(np.int64): "i64", np.dtype(np.bool_): "bool"}


def pack_message(fields: dict, topic: str = "pose", version: int = PROTOCOL_VERSION) -> bytes:
    """Byte-for-byte the layout of upstream gear_sonic/utils/teleop/zmq/zmq_planner_sender.py
    (pack_pose_message + _build_header), without importing the upstream package."""
    specs, payload = [], []
    for name, value in fields.items():
        value = np.ascontiguousarray(value)
        if value.dtype not in _DTYPES:
            raise TypeError(f"{name}: unsupported dtype {value.dtype}")
        value = value.astype(value.dtype.newbyteorder("<"), copy=False)
        specs.append({"name": name, "dtype": _DTYPES[value.dtype.newbyteorder("=")], "shape": list(value.shape)})
        payload.append(value.tobytes())
    header = json.dumps({"v": version, "endian": "le", "count": 1, "fields": specs},
                        separators=(",", ":")).encode("utf-8")
    if len(header) > HEADER_SIZE:
        raise ValueError(f"header too large: {len(header)} > {HEADER_SIZE}")
    return topic.encode("utf-8") + header.ljust(HEADER_SIZE, b"\x00") + b"".join(payload)


def unpack_message(message: bytes, topic: str = "pose") -> dict:
    """Inverse of pack_message (used by tests and by the sender's self-check)."""
    if not message.startswith(topic.encode()):
        raise ValueError("topic prefix mismatch")
    body = message[len(topic.encode()):]
    header = json.loads(body[:HEADER_SIZE].rstrip(b"\x00"))
    if header["endian"] != "le":
        raise ValueError("only little-endian payloads are produced here")
    inverse = {v: k for k, v in _DTYPES.items()}
    out, offset = {"_header": header}, HEADER_SIZE
    for spec in header["fields"]:
        dtype = inverse[spec["dtype"]].newbyteorder("<")
        count = int(np.prod(spec["shape"])) if spec["shape"] else 1
        out[spec["name"]] = np.frombuffer(body, dtype, count, offset).reshape(spec["shape"])
        offset += count * dtype.itemsize
    if offset != len(body):
        raise ValueError("trailing bytes after declared fields")
    return out


def mujoco_to_isaaclab():
    """Joint permutation from the upstream source (G1_MUJOCO_TO_ISAACLAB_DOF): isaac = mujoco[:, P]."""
    from ..export import mappings
    return np.asarray(mappings()["G1_MUJOCO_TO_ISAACLAB_DOF"], dtype=np.int64)


def forward_difference_velocity(q: np.ndarray, fps: float) -> np.ndarray:
    """Upstream convention (gear_sonic torch_humanoid_batch.py): v[i] = (q[i+1]-q[i])*fps, last = v[-2]."""
    v = np.empty_like(q)
    v[:-1] = (q[1:] - q[:-1]) * fps
    v[-1] = v[-2]
    return v


def stream_arrays(qpos50: np.ndarray, perm: np.ndarray | None = None) -> dict:
    """50 Hz (T,36) qpos [xyz, WXYZ, 29 MuJoCo joints] -> full-length Protocol v1 arrays."""
    q = np.asarray(qpos50, np.float64)
    if q.ndim != 2 or q.shape[1] != 7 + NUM_JOINTS or len(q) < 2:
        raise ValueError("expected (T,36) qpos with T>=2")
    perm = mujoco_to_isaaclab() if perm is None else perm
    joints = q[:, 7:][:, perm]
    quat = q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    return dict(joint_pos=joints.astype(np.float32),
                joint_vel=forward_difference_velocity(joints, CONTROL_FPS).astype(np.float32),
                body_quat=quat.astype(np.float32),
                frame_index=np.arange(len(q), dtype=np.int64))


def window_message(arrays: dict, start: int, length: int, *, catch_up: bool = True,
                   timestamp: float | None = None, topic: str = "pose") -> bytes:
    """Frames [start, start+length) of the full stream as one Protocol v1 message."""
    sl = slice(start, start + length)
    fields = {k: arrays[k][sl] for k in ("joint_pos", "joint_vel", "body_quat", "frame_index")}
    fields["catch_up"] = np.array([catch_up], dtype=np.bool_)
    if timestamp is not None:
        fields["timestamp_monotonic"] = np.array([timestamp], dtype=np.float64)
    return pack_message(fields, topic=topic)
