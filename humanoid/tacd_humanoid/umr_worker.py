"""UMR worker: runs inside the UMR environment (UMR_PYTHON; Python 3.12, torch 2.4.1+cu121).

Protocol: one JSON object per stdin line, one JSON reply per line on the original
stdout. UMR's own prints are redirected to stderr. Requests:
  {"op": "prepare"}                           build/train the G1 correspondence once
  {"op": "retarget", "data": npz, "out": npz} retarget one SMPL-X sequence
Only numpy/stdlib are imported here besides UMR itself.

adjacency_filtered_self_penetration_rows() contains a modified copy of
compute_robot_self_penetration_rows() from UMR (https://github.com/hanyang9/UMR, commit c56b630,
scripts/smpl_surface_retarget_common.py), used under the MIT License:

    Copyright (c) 2026 Hanyang Cao and UMR contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
    OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .paths import ASSETS, cache_dir, smplx_model_dir, umr_root

UMR = umr_root()
CORR = cache_dir() / "umr_correspondence"
# The base config is UMR's official G1 example with SMPL-X defaults; only the source template
# (one shared HY-fitted neutral shape), paths and viewer differ.
OVERRIDES_VERSION = "hy_g1_v1"
# Profile "tuned": robot/solver/retarget/correspondence settings from assets/umr_tuned/umr_tuned.json
# (collision-proxy30 XML, hard self-penetration, velocity filter, global_foot_joint ground alignment).
TUNED = ASSETS / "umr_tuned"
TUNED_CONFIG = TUNED / "umr_tuned.json"
TUNED_XML = TUNED / "g1_29dof_collisionproxy30.xml"
TUNED_OVERRIDES_VERSION = "hy_g1_tuned_v1"
PROFILES = ("tuned",)


def tuned_settings():
    return json.loads(TUNED_CONFIG.read_text())


def install_tuned_xml():
    """The proxy XML's meshdir is relative, so it is placed next to UMR's G1 meshes."""
    target = UMR / "assets/g1" / TUNED_XML.name
    if not target.exists() or target.read_bytes() != TUNED_XML.read_bytes():
        target.write_bytes(TUNED_XML.read_bytes())
    return target


def runtime_config(betas, profile="tuned", pose_init_iters=None):
    import numpy as np
    if profile not in PROFILES:
        raise ValueError(f"Unknown UMR profile {profile!r}; choose from {PROFILES}")
    betas = [float(x) for x in np.asarray(betas, dtype=np.float32).reshape(-1)[:10]]
    tag = hashlib.sha1(np.asarray(betas, dtype=np.float32).tobytes()).hexdigest()[:10]
    directory = CORR / f"g1_hy_{tag}"
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "extends": [str(UMR / "humanoid_retarget_defaults.json"),
                    str(UMR / "robot_configs/humanoid_retarget_unitree_g1_example.json")],
        "smplx_model_dir": str(smplx_model_dir()),
        "robot": {"xml": str(UMR / "assets/g1/g1_29dof.xml")},
        "smpl_template": {"source": "config", "type": "smplx", "gender": "neutral", "use_betas": True,
                          "use_gender": False, "name": "auto", "betas": betas},
        "correspondence": {"dataset": {"out": str(directory / "correspondence_dataset.npz")},
                           "train": {"out_dir": str(directory / "train")}},
        "motion": {"data": str(directory / "template_motion.npz"), "seq_key": "template_motion"},
        "retarget": {"out": str(directory / "unused.npz")},
        "view": {"enabled": False},
        "_t2a_overrides_version": OVERRIDES_VERSION,
    }
    name = "config.json"
    if profile == "tuned":
        tuned = tuned_settings()
        config["robot"] = {**tuned["robot"], "xml": str(install_tuned_xml())}
        config["solver"] = tuned["solver"]
        config["retarget"] = {**tuned["retarget"], "out": config["retarget"]["out"]}
        config["correspondence"] = {
            "dataset": {**tuned["correspondence"]["dataset"], **config["correspondence"]["dataset"]},
            "train": {**tuned["correspondence"]["train"], **config["correspondence"]["train"]},
            "slots_field": tuned["correspondence"]["slots_field"]}
        config["_t2a_overrides_version"] = TUNED_OVERRIDES_VERSION
        name = "config_tuned.json"
    if pose_init_iters is not None:
        # Iterations for the first solved frame only (later frames keep solver.iters). UMR starts
        # frame 0 from the zero pose with an L2 step cap, so a start pose far from zero needs more.
        config["solver"] = {**config.get("solver", {}), "pose_init_iters": int(pose_init_iters)}
        name = name.replace(".json", f"_pinit{int(pose_init_iters)}.json")
    path = directory / name
    text = json.dumps(config, indent=2, sort_keys=True)
    if not path.exists() or path.read_text() != text:
        # Atomic: parallel workers (shards) create the same file on first use and must never
        # read it half-written.
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    # A one-frame zero-pose motion with the shared betas; only consulted for metadata.
    template = directory / "template_motion.npz"
    if not template.exists():
        tmp = directory / f".template_motion.{os.getpid()}.npz"
        np.savez(tmp, poses=np.zeros((2, 55, 3), np.float32), trans=np.zeros((2, 3), np.float32),
                 betas=np.asarray(betas, np.float32), gender=np.asarray("neutral"),
                 mocap_frame_rate=np.asarray(30), output_up=np.asarray("z"))
        os.replace(tmp, template)
    return path, directory


def adjacency_filtered_self_penetration_rows(common):
    """UMR's compute_robot_self_penetration_rows() never calls is_adjacent_body_pair(), so literal
    parent/child joint-interface overlaps enter the hard constraint. Only the
    `if is_adjacent_body_pair(...): continue` lines differ from upstream (MIT, see module docstring)."""
    import mujoco
    import numpy as np

    def is_adjacent_body_pair(model, body1, body2):
        body1 = int(body1)
        body2 = int(body2)
        if body1 == body2:
            return True
        parent = model.body_parentid
        return int(parent[body1]) == body2 or int(parent[body2]) == body1

    def compute_robot_self_penetration_rows(model, data, cache, collision_threshold):
        if cache is None:
            return [], []
        robot_geoms = np.asarray(cache["geom_ids"], dtype=np.int32)
        if robot_geoms.size == 0:
            return [], []
        threshold = float(collision_threshold)
        labels = cache["labels"]
        saved_margin = model.geom_margin.copy()
        candidates = set()
        try:
            model.geom_margin[:] = np.maximum(saved_margin, threshold)
            mujoco.mj_collision(model, data)
            robot_geom_set = set(int(g) for g in robot_geoms)
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                if geom1 < 0 or geom2 < 0:
                    continue
                if geom1 in robot_geom_set and geom2 in robot_geom_set:
                    body1 = int(model.geom_bodyid[geom1])
                    body2 = int(model.geom_bodyid[geom2])
                    if is_adjacent_body_pair(model, body1, body2):
                        continue
                    candidates.add((min(geom1, geom2), max(geom1, geom2)))
        finally:
            model.geom_margin[:] = saved_margin
        jacobians = []
        distances = []
        fromto = np.zeros(6, dtype=np.float64)
        for geom1, geom2 in sorted(candidates):
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(model, data, geom1, geom2, threshold, fromto)
            if dist <= threshold:
                jacobians.append(common.collision_relative_jacobian(model, data, geom1, geom2, labels, fromto, dist))
                distances.append(float(dist))
        return jacobians, distances

    return compute_robot_self_penetration_rows


def install_exact_caches(common):
    """Memoize clip-independent UMR work inside the persistent worker. Every cached function is a
    pure function of its (hashed) inputs and callers get fresh copies, so outputs are bitwise
    unchanged."""
    import collections
    import copy
    import hashlib
    import types
    import numpy as np
    import smplx.body_models

    def digest(*parts):
        h = hashlib.sha1()
        for part in parts:
            if isinstance(part, np.ndarray):
                h.update(str((part.dtype, part.shape)).encode()); h.update(np.ascontiguousarray(part).tobytes())
            else:
                h.update(json.dumps(part, sort_keys=True, default=repr).encode())
        return h.hexdigest()

    def lru(size):
        store = collections.OrderedDict()
        def get(key, compute):
            if key in store:
                store.move_to_end(key)
            else:
                store[key] = compute()
                while len(store) > size:
                    store.popitem(last=False)
            return copy.deepcopy(store[key])
        return get

    # Surface binding of the (shared) template and robot meshes: ~4 s per clip.
    bind, bind_cache = common.bind_points_to_mesh, lru(8)
    def bind_points_to_mesh(points, vertices, faces, nearest_vertex_k=24):
        key = digest(np.asarray(points, np.float32), np.asarray(vertices, np.float32),
                     np.asarray(faces, np.int32), int(nearest_vertex_k))
        return bind_cache(key, lambda: bind(points, vertices, faces, nearest_vertex_k))
    common.bind_points_to_mesh = bind_points_to_mesh

    # Zero-pose template vertices of the shared shape: ~0.9 s per clip.
    template, template_cache = common.source_template_vertices_joints_faces, lru(4)
    def source_template_vertices_joints_faces(sequence, template_cfg, smplx_model_dir, soma_usd_path=None):
        if common.source_model_type(sequence, template_cfg) != "smplx":
            return template(sequence, template_cfg, smplx_model_dir, soma_usd_path)
        gender = str(template_cfg.get("gender", sequence.get("gender", "neutral"))).lower()
        key = digest(str(Path(smplx_model_dir).resolve()), gender, template_cfg.get("betas"))
        return template_cache(key, lambda: template(sequence, template_cfg, smplx_model_dir, soma_usd_path))
    common.source_template_vertices_joints_faces = source_template_vertices_joints_faces

    # SMPL-X model files: smplx re-reads and decompresses the npz for every model it builds
    # (one per batch size, several per clip). Serve the arrays from memory instead.
    files = {}
    def load(path, *args, **kwargs):
        key = str(Path(path).resolve()) if isinstance(path, (str, Path)) else None
        if key is None or not key.endswith(".npz") or not Path(key).name.startswith("SMPLX_"):
            return np.load(path, *args, **kwargs)
        if key not in files:
            with np.load(path, *args, **kwargs) as data:
                files[key] = {k: data[k] for k in data.files}
        return {k: v.copy() for k, v in files[key].items()}
    proxy = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    proxy.load = load
    smplx.body_models.np = proxy


def cone_slack_violation(A, b, cone_dims, x):
    """Largest violation of s = b - A x in the Clarabel cones [(kind, dim)], kind in {"nonneg", "soc"}."""
    import numpy as np
    s = np.asarray(b, dtype=np.float64) - A @ np.asarray(x, dtype=np.float64)
    worst, row = 0.0, 0
    for kind, dim in cone_dims:
        part = s[row:row + dim]
        if dim:
            worst = max(worst, float(np.maximum(0.0, -part).max()) if kind == "nonneg"
                        else float(max(0.0, np.linalg.norm(part[1:]) - part[0])))
        row += dim
    return worst


def accept_insufficient_progress(clarabel, stats, tolerance=1e-6):
    """A Clarabel namespace whose DefaultSolver reports InsufficientProgress as AlmostSolved when the
    last iterate is finite, feasible within `tolerance` and no worse than the zero step.

    UMR raises on any status other than Solved/AlmostSolved, which kills the clip. Captured
    InsufficientProgress QPs had a feasible iterate within 4.7e-6 of the optimum found with loosened tolerances: the interior-point gap had stalled at the
    optimum, not failed. Whether a clip meets such a QP depends on float paths that vary between runs,
    so without this a clip can succeed in one run and fail in the next. Other statuses still raise;
    QPs that solve normally are untouched (bitwise)."""
    import re
    import types
    import numpy as np

    def dims(cones):
        out = []
        for cone in cones:
            name, dim = re.match(r"(\w+)\((\d+)\)", repr(cone)).groups()
            if name not in ("NonnegativeConeT", "SecondOrderConeT"):
                return None
            out.append(("nonneg" if name == "NonnegativeConeT" else "soc", int(dim)))
        return out

    class Solver:  # composition: the Rust class cannot be subclassed
        def __init__(self, P, q, A, b, cones, settings):
            self.problem = (P, np.asarray(q, dtype=np.float64), A, np.asarray(b, dtype=np.float64), dims(cones))
            self.inner = clarabel.DefaultSolver(P, q, A, b, cones, settings)

        def solve(self):
            result = self.inner.solve()
            if str(result.status) != "InsufficientProgress":
                return result
            P, q, A, b, cone_dims = self.problem
            x = np.asarray(result.x, dtype=np.float64)
            # Clarabel reads only the upper triangle of P.
            quadratic = x @ (P @ x) + x @ (P.T @ x) - x @ (P.diagonal() * x)
            ok = bool(cone_dims and np.isfinite(x).all()
                      and cone_slack_violation(A, b, cone_dims, x) <= tolerance
                      and 0.5 * quadratic + q @ x <= 0.0)
            stats["insufficient_progress"] = stats.get("insufficient_progress", 0) + 1
            if not ok:
                stats["insufficient_progress_rejected"] = stats.get("insufficient_progress_rejected", 0) + 1
                return result
            stats["insufficient_progress_accepted"] = stats.get("insufficient_progress_accepted", 0) + 1
            return types.SimpleNamespace(status="AlmostSolved", x=result.x, iterations=result.iterations,
                                         obj_val=getattr(result, "obj_val", None))

    namespace = types.SimpleNamespace(**{k: getattr(clarabel, k) for k in dir(clarabel) if not k.startswith("__")})
    namespace.DefaultSolver = Solver
    return namespace


class Worker:
    def __init__(self, betas, profile="tuned", adjacency_fix=None, exact_caches=True, pose_init_iters=None):
        sys.path[:0] = [str(UMR), str(UMR / "scripts")]
        os.chdir(UMR)
        import humanoid_retarget_pipeline as pipeline
        import smpl_surface_retarget_common as common
        self.pipeline = pipeline
        self.profile = profile
        self.adjacency_fix = profile == "tuned" if adjacency_fix is None else bool(adjacency_fix)
        if self.adjacency_fix:
            common.compute_robot_self_penetration_rows = adjacency_filtered_self_penetration_rows(common)
        self.exact_caches = bool(exact_caches)
        if self.exact_caches:
            install_exact_caches(common)
        self.qp_stats = {}
        common.clarabel = accept_insufficient_progress(common.clarabel, self.qp_stats)
        self.pose_init_iters = pose_init_iters
        self.config_path, self.directory = runtime_config(betas, profile, pose_init_iters)
        self.config = pipeline.load_pipeline_config(self.config_path)

    def prepare(self):
        started = time.perf_counter()
        dataset = self.pipeline.build_correspondence_dataset(self.config)
        slots = self.pipeline.train_correspondence(self.config, dataset)
        if not self.pipeline.correspondence_slots_compatible(slots, self.config):
            raise RuntimeError(f"Correspondence slots incompatible after training: {slots}")
        self.slots = slots
        return dict(slots=str(slots), dataset=str(dataset), seconds=time.perf_counter() - started,
                    template=self.pipeline.expected_smpl_slot_name(self.config), config=str(self.config_path))

    def retarget(self, data, out):
        import retarget_smpl_to_humanoid_surface_vector as script
        import numpy as np
        if not hasattr(self, "slots"):
            self.prepare()
        solver = self.pipeline.section(self.config, "solver")
        args = {key: solver[key] for key in self.pipeline.RETARGET_SOLVER_ARGS if key in solver}
        # The data file holds one sequence keyed by its stem; name it explicitly because the
        # config's motion.seq_key (the template placeholder) would otherwise be used.
        argv = ["retarget", "--config", str(self.config_path), "--data", str(data), "--seq-key", Path(data).stem,
                "--slots", str(self.slots), "--out", str(out),
                "--slots-field", str(self.pipeline.section(self.config, "correspondence").get("slots_field", "reconstructed_slots")),
                *self.pipeline.list_of_args(args)]
        started = time.perf_counter()
        self.qp_stats.clear()
        # UMR writes a per-clip floating-root copy of the robot XML next to the meshes; remove it even when the clip fails.
        floating = Path(self.pipeline.section(self.config, "robot")["xml"]).parent / Path(out).with_suffix(".floating_mjcf.xml").name
        old = sys.argv
        sys.argv = argv
        try:
            script.main()
        finally:
            sys.argv = old
            floating.unlink(missing_ok=True)
        seconds = time.perf_counter() - started
        with np.load(out, allow_pickle=True) as result:
            names = [str(x) for x in result["robot_joint_names"]]
            shape = list(result["qpos"].shape)
        return dict(out=str(out), seconds=seconds, robot_joint_names=names, qpos_shape=shape, argv=argv[1:],
                    profile=self.profile, adjacency_fix=self.adjacency_fix, exact_caches=self.exact_caches,
                    qp_insufficient_progress_accepted=self.qp_stats.get("insufficient_progress_accepted", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas-json", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="tuned")
    parser.add_argument("--adjacency-fix", choices=("auto", "on", "off"), default="auto",
                        help="auto: on for tuned")
    parser.add_argument("--no-exact-caches", action="store_true", help="recompute clip-independent work per clip")
    parser.add_argument("--pose-init-iters", type=int, default=None, help="first-frame iterations (default: profile config)")
    args = parser.parse_args()
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    betas = json.loads(Path(args.betas_json).read_text())["betas"]
    worker = Worker(betas, args.profile, None if args.adjacency_fix == "auto" else args.adjacency_fix == "on",
                    not args.no_exact_caches, args.pose_init_iters)
    import torch
    replies.write(json.dumps(dict(ready=True, profile=worker.profile, adjacency_fix=worker.adjacency_fix,
                                  exact_caches=worker.exact_caches,
                                  pose_init_iters=int(worker.pipeline.section(worker.config, "solver").get("pose_init_iters", -1)),
                                  config=str(worker.config_path), torch=torch.__version__, cuda=torch.cuda.is_available(),
                                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            if request["op"] == "prepare":
                reply = worker.prepare()
            elif request["op"] == "retarget":
                reply = worker.retarget(request["data"], request["out"])
            elif request["op"] == "exit":
                break
            else:
                raise ValueError(f"Unknown op {request['op']}")
            reply["ok"] = True
        except Exception as error:  # Report and keep serving; the caller decides.
            reply = dict(ok=False, error=repr(error), traceback=traceback.format_exc())
        replies.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
