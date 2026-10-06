"""HY human motion -> Unitree G1 with UMR (tuned profile) and a per-frame root-Z ground fix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Protocol

import numpy as np

from .motion import HumanMotion, RobotMotion
from .paths import PACKAGE, cache_dir, umr_python

class Retargeter(Protocol):
    def retarget(self, motion: HumanMotion) -> RobotMotion: ...


UMR_SOURCE_COMMIT = "c56b6301ded02a187a30cc6aafa4f535735104d2"
G1_MUJOCO_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"]


GROUND_FIX_MARGIN_M = 1e-6


_GROUND_FIX_MODEL = None


def _ground_fix_model():
    """Box-foot G1 plus, per collidable mesh, only its convex-hull vertices: the lowest point
    of a mesh along any direction is a hull vertex, so the minimum is unchanged (checked
    bitwise against geometry.geom_floor on 140 clips) at ~1/10 of the dot products."""
    global _GROUND_FIX_MODEL
    if _GROUND_FIX_MODEL is None:
        import mujoco
        from scipy.spatial import ConvexHull
        from .transition import box_feet_model
        model = box_feet_model()
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
        hulls = {}
        for g in geoms:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start + count]
                hulls[g] = np.ascontiguousarray(vertices[np.sort(ConvexHull(vertices).vertices)])
        _GROUND_FIX_MODEL = model, geoms, hulls
    return _GROUND_FIX_MODEL


def ground_fix(qpos, margin=GROUND_FIX_MARGIN_M):
    """Per frame, raise the root only where the robot penetrates the
    floor, leaving it `margin` below z=0 (lift_t = max(0, -lowest_t - margin)). Root x/y,
    orientation and joints are untouched. Measured on the deployment box-foot G1."""
    import mujoco
    from .geometry import geom_floor
    model, geoms, hulls = _ground_fix_model()
    data = mujoco.MjData(model)

    def floor(g):
        if g in hulls:
            return float(data.geom_xpos[g, 2] + (hulls[g] @ data.geom_xmat[g].reshape(3, 3)[2]).min())
        return geom_floor(model, data, g)

    lift = np.zeros(len(qpos))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        lift[t] = max(0., -min(floor(g) for g in geoms) - margin)
    out = np.array(qpos, dtype=np.float64, copy=True)
    out[:, 2] += lift
    return out, lift


def umr_qpos_to_mujoco(qpos, joint_names):
    """Map UMR's (T, 7+n) qpos to our 29-DoF MuJoCo order strictly by joint name."""
    qpos = np.asarray(qpos, dtype=np.float64)
    names = [str(n) for n in joint_names]
    hinge = [n for n in names if n in G1_MUJOCO_JOINTS]
    if len(hinge) != len(set(hinge)) or set(hinge) != set(G1_MUJOCO_JOINTS):
        raise ValueError(f"UMR joints do not cover the 29 G1 joints exactly: {names}")
    # UMR lists every model joint; the free joint (first, unnamed or named) owns qpos[:7].
    offset = qpos.shape[1] - len(hinge)
    if offset != 7:
        raise ValueError(f"UMR qpos width {qpos.shape[1]} does not equal 7 + {len(hinge)} joints")
    index = {n: offset + i for i, n in enumerate(hinge)}
    out = np.concatenate([qpos[:, :7], qpos[:, [index[n] for n in G1_MUJOCO_JOINTS]]], axis=1)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


UMR_PROFILES = {
    # name: (worker profile, default bridge, root-Z ground fix, first-frame iterations)
    # Frame 0 is solved with 60 iterations: from the zero pose under a 0.4 rad L2 step cap, the
    # profile's 15 leave raised-arm / jump starts ~1-1.4 rad short; 60 reach the 1000-iteration
    # pose within 5 mrad.
    "umr_tuned_f0": ("tuned", "fit", True, 60),
}
UMR_BRIDGES = ("fit", "direct")


class UMRRetargeter:
    """Surface-correspondence retargeting (UMR) of the HY body via a SMPL-X surface.

    The SMPL-X bridge runs in this process (torch); UMR runs in a persistent worker inside its
    own environment (UMR_PYTHON) so its torch 2.4.1/cu121 stack stays isolated.
    "umr_tuned_f0": tuned UMR settings (collision-proxy30 XML, adjacency fix, hard
    self-penetration, velocity filter, global_foot_joint ground alignment), 60 first-frame
    iterations, plus the per-frame root-Z ground fix.
    bridge "fit": shared betas + per-clip residual pose fit to HY joints; "direct": HY local
    rotations drive SMPL-X with the shared betas, only the pelvis translation is matched.
    """

    def __init__(self, device=None, worker_device=None, name="umr_tuned_f0", bridge=None, adjacency_fix="auto"):
        if name not in UMR_PROFILES:
            raise ValueError(f"Unknown UMR retargeter {name!r}; choose from {tuple(UMR_PROFILES)}")
        self.name = name
        self.profile, default_bridge, self.ground_fix, self.pose_init_iters = UMR_PROFILES[name]
        self.bridge = bridge or default_bridge
        if self.bridge not in UMR_BRIDGES:
            raise ValueError(f"Unknown UMR bridge {self.bridge!r}; choose from {UMR_BRIDGES}")
        # Later stages (ground fix, render) import mujoco in this process and need the headless
        # EGL backend chosen before the first import.
        os.environ.setdefault("MUJOCO_GL", "egl")
        import torch
        from .smplx_bridge import BETAS_FILE
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.betas_file = BETAS_FILE
        scratch_root = cache_dir() / "umr_scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.scratch = Path(tempfile.mkdtemp(prefix="umr_", dir=scratch_root))
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(PACKAGE.parent),
                   MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
        if worker_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(worker_device)
        self.log = open(self.scratch / "worker.log", "a")
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [umr_python(), "-m", "tacd_humanoid.umr_worker", "--betas-json", str(BETAS_FILE),
             "--profile", self.profile, "--adjacency-fix", adjacency_fix,
             *([] if self.pose_init_iters is None else ["--pose-init-iters", str(self.pose_init_iters)])],
            cwd=PACKAGE.parent, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.worker_info = self._read()
        self.prepared = self._call(dict(op="prepare"))
        self.startup_seconds = time.perf_counter() - started

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"UMR worker exited; see {self.scratch / 'worker.log'}")
        return json.loads(line)

    def _call(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        reply = self._read()
        if not reply.get("ok", False):
            raise RuntimeError(f"UMR worker failed: {reply.get('error')}\n{reply.get('traceback', '')}")
        return reply

    def _bridge(self, motion):
        """SMPL-X bridge for one clip -> job dict (the worker request is not sent yet)."""
        from .smplx_bridge import human_to_smplx, write_umr_npz
        started = time.perf_counter()
        params = human_to_smplx(motion, device=self.device, fit=self.bridge == "fit")
        bridge_seconds = time.perf_counter() - started
        stem = f"clip_{time.time_ns()}"
        data, out = self.scratch / f"{stem}.npz", self.scratch / f"{stem}_g1.npz"
        write_umr_npz(params, data)
        return dict(motion=motion, params=params, data=data, out=out, bridge_seconds=bridge_seconds, started=started)

    def _send(self, job):
        self.process.stdin.write(json.dumps(dict(op="retarget", data=str(job["data"]), out=str(job["out"]))) + "\n")
        self.process.stdin.flush()

    def _receive(self):
        reply = self._read()
        if not reply.get("ok", False):
            raise RuntimeError(f"UMR worker failed: {reply.get('error')}\n{reply.get('traceback', '')}")
        return reply

    def retarget(self, motion):
        job = self._bridge(motion)
        self._send(job)
        return self._finish(job, self._receive())

    def retarget_many(self, motions):
        """Same results as retarget() one by one (bitwise), but the bridge of clip i+1 runs in
        this process while the worker solves clip i. Yields (index, RobotMotion or exception)."""
        motions = list(motions)
        if not motions:
            return
        job = self._bridge(motions[0]); self._send(job)
        for i in range(len(motions)):
            upcoming = self._bridge(motions[i + 1]) if i + 1 < len(motions) else None
            try:
                result = self._finish(job, self._receive(), pipelined=True)
            except RuntimeError as error:  # worker stays alive after a failed clip
                for path in (job["data"], job["out"]):
                    path.unlink(missing_ok=True)
                result = error
            yield i, result
            if upcoming is not None:
                job = upcoming; self._send(job)

    def _finish(self, job, reply, pipelined=False):
        motion, params, data, out = job["motion"], job["params"], job["data"], job["out"]
        bridge_seconds, started = job["bridge_seconds"], job["started"]
        with np.load(out, allow_pickle=True) as result:
            raw = np.asarray(result["qpos"])
            names = [str(n) for n in result["robot_joint_names"]]
            fps = float(np.asarray(result["fps"]).reshape(-1)[0])
            frame_ids = np.asarray(result["frame_ids"])
        if fps != motion.fps or len(raw) != len(motion.transl) or not np.array_equal(frame_ids, np.arange(len(raw))):
            raise ValueError(f"UMR returned {len(raw)} frames at {fps} fps for a {len(motion.transl)}-frame clip")
        qpos = umr_qpos_to_mujoco(raw, names)
        extra = {}
        if self.ground_fix:
            fix_started = time.perf_counter()
            qpos, lift = ground_fix(qpos)
            extra = dict(ground_fix_margin_m=GROUND_FIX_MARGIN_M, ground_fix_max_lift_m=float(lift.max()),
                         ground_fix_lifted_frames=int((lift > 0).sum()),
                         ground_fix_seconds=time.perf_counter() - fix_started)
        validation = {k: v for k, v in params["validation"].items() if k != "per_joint"}
        robot = RobotMotion(qpos, metadata={**motion.metadata, "retargeter": self.name,
            "umr_profile": self.profile, "umr_bridge": self.bridge,
            "umr_adjacency_fix": self.worker_info.get("adjacency_fix"),
            "umr_pose_init_iters": self.worker_info.get("pose_init_iters"), **extra,
            "umr_source_commit": UMR_SOURCE_COMMIT, "umr_template": self.prepared["template"],
            "umr_slots": self.prepared["slots"], "smplx_fit": validation,
            "umr_solver_seconds": reply["seconds"], "smplx_bridge_seconds": bridge_seconds,
            "umr_qp_insufficient_progress_accepted": reply.get("qp_insufficient_progress_accepted"),
            "smplx_bridge_device": self.device, "umr_pipelined": pipelined,
            "retarget_seconds": time.perf_counter() - started})
        robot.validate()
        for path in (data, out):
            path.unlink(missing_ok=True)
        return robot

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps(dict(op="exit")) + "\n")
                self.process.stdin.close()
                self.process.wait(timeout=60)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
        self.process.stdout.close()
        self.log.close()

    def __del__(self):
        if hasattr(self, "process"):
            self.close()


RETARGETERS = tuple(UMR_PROFILES)
DEFAULT_RETARGETER = "umr_tuned_f0"


def make_retargeter(name=DEFAULT_RETARGETER, **kwargs):
    if name in UMR_PROFILES:
        return UMRRetargeter(name=name, **kwargs)
    raise ValueError(f"Unknown retargeter {name!r}; choose from {RETARGETERS}")


def add_retargeter_argument(parser):
    parser.add_argument("--retargeter", choices=RETARGETERS, default=DEFAULT_RETARGETER,
                        help="Human-to-G1 retargeting backend (default %(default)s)")


def retarget(motion, backend: Retargeter | None = None):
    return (backend or make_retargeter()).retarget(motion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", required=True)
    parser.add_argument("--out", required=True)
    add_retargeter_argument(parser)
    parser.add_argument("--umr-bridge", choices=UMR_BRIDGES, default=None, help="default per profile")
    args = parser.parse_args()
    backend = make_retargeter(args.retargeter, bridge=args.umr_bridge)
    robot = retarget(HumanMotion.load(args.human), backend)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    robot.save(out / f"{args.retargeter}.npz")
    print(json.dumps(robot.metadata, indent=2))


if __name__ == "__main__":
    main()
