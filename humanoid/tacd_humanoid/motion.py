"""Stage contracts: meters/radians, explicit coordinates, quaternion and joint order."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass
class HumanMotion:
    rot6d: np.ndarray  # (T,22,6), local HY rotations, y-up
    transl: np.ndarray  # (T,3), smoothed and ground-aligned once
    joints_world: np.ndarray  # (T,22,3), FK + aligned translation, y-up
    fps: float = 30.0
    ground_offset: float = 0.0
    metadata: dict = field(default_factory=dict)

    def validate(self):
        t = len(self.transl)
        if self.rot6d.shape != (t, 22, 6) or self.transl.shape != (t, 3) or self.joints_world.shape != (t, 22, 3):
            raise ValueError("HumanMotion must have aligned (T,22,6), (T,3), (T,22,3) arrays")
        if t < 2 or self.fps != 30 or not all(np.isfinite(v).all() for v in [self.rot6d, self.transl, self.joints_world]):
            raise ValueError("HumanMotion must be finite, >=2 frames, at 30 fps")

    def save(self, path):
        self.validate()
        np.savez_compressed(path, rot6d=self.rot6d, transl=self.transl, joints_world=self.joints_world,
                            fps=self.fps, ground_offset=self.ground_offset, metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            value = cls(d["rot6d"], d["transl"], d["joints_world"], float(d["fps"]),
                        float(d["ground_offset"]), json.loads(str(d["metadata"])))
        value.validate()
        return value


@dataclass
class RobotMotion:
    qpos: np.ndarray  # (T,36): xyz, quaternion WXYZ, MuJoCo 29 joint order; z-up
    fps: float = 30.0
    metadata: dict = field(default_factory=dict)
    source_qpos: np.ndarray | None = None  # Original motion for eased-time evaluation.

    def validate(self):
        q = self.qpos
        if q.ndim != 2 or q.shape[1] != 36 or len(q) < 2 or not np.isfinite(q).all():
            raise ValueError("RobotMotion qpos must be finite (T,36), T>=2")
        if self.fps != 30.0 or not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-3):
            raise ValueError("RobotMotion must be 30 fps with unit WXYZ root quaternions")
        if self.source_qpos is not None:
            RobotMotion(self.source_qpos, self.fps).validate()

    def save(self, path):
        self.validate()
        source = {} if self.source_qpos is None else dict(source_qpos=self.source_qpos)
        np.savez_compressed(path, qpos=self.qpos, fps=self.fps, joint_order="mujoco",
                            quaternion_order="wxyz", metadata=json.dumps(self.metadata), **source)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if str(d["joint_order"]) != "mujoco" or str(d["quaternion_order"]) != "wxyz":
                raise ValueError("Unexpected qpos coordinate convention")
            value = cls(d["qpos"], float(d["fps"]), json.loads(str(d["metadata"])),
                        d["source_qpos"] if "source_qpos" in d else None)
        value.validate()
        return value
