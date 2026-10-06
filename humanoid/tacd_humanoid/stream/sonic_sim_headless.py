"""Headless launcher for SONIC's own MuJoCo sim2sim process (gear_sonic/scripts/run_sim_loop.py).

Runs the unmodified upstream simulator (BaseSimulator / DefaultEnv, scene_43dof, 200 Hz physics,
Unitree SDK2 DDS bridge on `lo`) without a viewer. Two things the upstream entry point only offers
through the GLFW viewer are provided here instead:

* operator keys on stdin, one per line: ``9`` toggles the elastic band exactly like pressing 9 in the
  viewer (``DefaultEnv.handle_keyboard_button``), ``7``/``8`` lower/raise it by 0.1 m like the viewer keys, ``backspace`` resets the sim; ``rec_start NAME`` /
  ``rec_stop`` record the simulated robot state; ``quit`` exits.
* a recorder that samples the MuJoCo state every 4th physics step (50 Hz) into
  ``<out_dir>/<NAME>.npz``: sim time, wall time, root xyz + WXYZ quaternion, 29 body joints in MuJoCo
  order (same layout as RobotMotion qpos), joint velocities, band state and the upstream fall flag.

The physics loop body is the same as ``BaseSimulator.start`` (sim_step, reward cadence); the recorder
and the stdin command poll are added.  One deliberate difference: upstream's rate limiter sleeps
``dt - elapsed`` per step and never catches up, so on a loaded shared host sim time falls behind the
deploy's wall-clock 50 Hz loop (we measured 125-275 late steps before release and the robot fell on
release; with 2 late steps it stood).  By default this loop keeps an absolute schedule (sim time tracks
wall time, catching up after hiccups, resync if more than --max-lag-s behind); --upstream-rate-limiter
restores the upstream behaviour.

Second deliberate difference: the upstream bridge writes five DDS topics per physics step (rt/lowstate,
rt/odostate, rt/secondary_imu, rt/dex3/{left,right}/state).  Python cyclonedds serialization of these
is ~83% of the step time (cProfile: 4.7 s of 5.6 s over 2000 steps; mj_step 0.4 s), which is what pushes
the step past its 5 ms budget on a loaded host.  The deploy (g1_deploy_onnx_ref) subscribes only to
rt/lowstate and rt/secondary_imu, so by default the three unused topics are not written
(--all-dds-topics restores them).  The published content of the two used topics is unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import select
import signal
import sys
import threading
import time

import numpy as np


def _stdin_reader(commands: "queue.Queue[str]", stop: threading.Event):
    fd = sys.stdin.fileno()
    buffer = b""
    while not stop.is_set():
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            time.sleep(0.2)  # writer closed; keep polling (stdin is normally an O_RDWR fifo)
            continue
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.decode(errors="replace").strip()
            if line:
                commands.put(line)


def _atomic_npz(path: Path, **arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    with np.load(tmp, allow_pickle=False) as check:  # read back before publishing the file
        assert set(check.files) == set(arrays)
    os.replace(tmp, path)


class Recorder:
    def __init__(self, env, out_dir: Path):
        self.env, self.out_dir = env, out_dir
        self.name, self.rows, self.late, self.diag = None, [], [], []
        model = env.mj_model
        self.joint_qpos = np.array([model.jnt_qposadr[j] for j in env.body_joint_index])
        self.joint_qvel = np.array([model.jnt_dofadr[j] for j in env.body_joint_index])
        self.joint_names = [model.joint(int(j)).name for j in env.body_joint_index]

    def start(self, name):
        self.name, self.rows, self.late, self.diag = name, [], [], []
        print(f"[sim_headless] recording '{name}'", flush=True)

    def sample(self, wall, late_steps=0, cmd_age=np.nan, step_ms=np.nan):
        if self.name is None:
            return
        d = self.env.mj_data
        band = getattr(self.env, "elastic_band", None)
        self.late.append(late_steps)
        self.diag.append((cmd_age, step_ms))
        self.rows.append(np.concatenate([
            [d.time, wall, float(band.enable) if band is not None else 0.0, float(self.env.fall)],
            d.qpos[:7], d.qpos[self.joint_qpos], d.qvel[:6], d.qvel[self.joint_qvel]]))

    def stop(self):
        if self.name is None:
            return None
        rows = np.asarray(self.rows) if self.rows else np.zeros((0, 4 + 36 + 35))
        path = self.out_dir / f"{self.name}.npz"
        _atomic_npz(path, sim_time=rows[:, 0], wall_time=rows[:, 1], band_enabled=rows[:, 2],
                    fall_flag=rows[:, 3], qpos=rows[:, 4:40], qvel=rows[:, 40:75], fps=np.float64(50.0),
                    joint_order=np.str_("mujoco"), quaternion_order=np.str_("wxyz"),
                    joint_names=np.array(self.joint_names), late_steps=np.asarray(self.late, dtype=np.int64),
                    lowcmd_age_s=np.asarray([a for a, _ in self.diag]), sim_step_ms=np.asarray([b for _, b in self.diag]))
        print(f"[sim_headless] saved {len(rows)} samples -> {path}", flush=True)
        self.name, self.rows, self.late, self.diag = None, [], [], []
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--upstream-rate-limiter", action="store_true",
                        help="use upstream's per-step sleep (drifts behind real time under load)")
    parser.add_argument("--max-lag-s", type=float, default=0.1)
    parser.add_argument("--all-dds-topics", action="store_true",
                        help="also write rt/odostate and rt/dex3/*/state every step (unused by the deploy)")
    parser.add_argument("--record-from-start", default=None, help="start recording NAME immediately")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so --help works without the SONIC sim environment.
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

    config = SimLoopConfig(interface=args.interface, enable_onscreen=False, enable_offscreen=False)
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    instantiate_g1_robot_model()  # same model instantiation as run_sim_loop.py
    init_channel(config=wbc_config)
    sim = SimulatorFactory.create_simulator(config=wbc_config, env_name=config.env_name,
                                            onscreen=False, offscreen=False, enable_image_publish=False)
    env = sim.sim_env
    bridge = getattr(env, "unitree_bridge", None) or getattr(sim, "unitree_bridge", None)
    if not args.all_dds_topics:
        class _Unpublished:  # the deploy never subscribes to these topics
            def Write(self, *_a, **_k):
                return True
        for name in ("odo_state_puber", "left_hand_state_puber", "right_hand_state_puber"):
            if getattr(bridge, name, None) is not None:
                setattr(bridge, name, _Unpublished())
        print("[sim_headless] DDS: writing rt/lowstate + rt/secondary_imu only", flush=True)
    rec = Recorder(env, out_dir)
    if args.record_from_start:
        rec.start(args.record_from_start)
    env.fall = False
    print("[sim_headless] config:", json.dumps({k: wbc_config[k] for k in
          ["ROBOT_SCENE", "INTERFACE", "DOMAIN_ID", "SIMULATE_DT", "ENABLE_ELASTIC_BAND", "FREE_BASE", "USE_SENSOR"]}),
          flush=True)

    commands: "queue.Queue[str]" = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=_stdin_reader, args=(commands, stop), daemon=True).start()
    running = [True]

    def _terminate(*_):
        running[0] = False
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)

    sim_dt = sim.sim_dt
    steps_per_sample = int(round(0.02 / sim_dt))
    sim_cnt, late_steps = 0, 0
    t_start = time.monotonic()
    t_sched = t_start
    last_cmd, last_cmd_time = getattr(bridge, "low_cmd", None), np.nan
    resyncs = 0
    falls = 0
    try:
        while running[0]:
            while not commands.empty():
                cmd = commands.get().split()
                if cmd[0] == "9":
                    env.handle_keyboard_button("9")
                elif cmd[0] in ("7", "8") and getattr(env, "elastic_band", None) is not None:
                    # same as the viewer's key callback (ElasticBand.MujuocoKeyCallback): 7 lowers, 8 raises
                    env.elastic_band.length += -0.1 if cmd[0] == "7" else 0.1
                    print(f"[sim_headless] elastic band length {env.elastic_band.length:+.1f}", flush=True)
                elif cmd[0] == "backspace":
                    env.handle_keyboard_button("backspace")
                elif cmd[0] == "rec_start" and len(cmd) == 2:
                    rec.start(cmd[1])
                elif cmd[0] == "rec_stop":
                    rec.stop()
                elif cmd[0] == "status":
                    band = getattr(env, "elastic_band", None)
                    print(f"[sim_headless] STATUS id={cmd[1] if len(cmd) > 1 else '-'} falls={falls} "
                          f"pelvis_z={env.mj_data.qpos[2]:.3f} late_steps={late_steps} resyncs={resyncs} "
                          f"band={int(bool(band and band.enable))}", flush=True)
                elif cmd[0] == "quit":
                    running[0] = False
                else:
                    print(f"[sim_headless] unknown command {cmd}", flush=True)
            step_start = time.monotonic()
            cmd = getattr(bridge, "low_cmd", None)
            if cmd is not last_cmd:  # LowCmdHandler replaced the message object: a new command arrived
                last_cmd, last_cmd_time = cmd, step_start
            env.sim_step()  # upstream: PD torques from DDS low_cmd, mj_step, check_fall
            step_ms = (time.monotonic() - step_start) * 1e3
            if env.fall:
                falls += 1
            if sim_cnt % int(sim.reward_dt / sim_dt) == 0:
                env.update_reward()
            if sim_cnt % steps_per_sample == 0:
                rec.sample(step_start, late_steps, step_start - last_cmd_time, step_ms)
            if sim_cnt % (steps_per_sample * 250) == 0:
                elapsed = time.monotonic() - t_start
                print(f"[sim_headless] sim_t={env.mj_data.time:.2f}s wall={elapsed:.2f}s "
                      f"rtf={env.mj_data.time / max(elapsed, 1e-9):.3f} late_steps={late_steps} "
                      f"pelvis_z={env.mj_data.qpos[2]:.3f} falls={falls} resyncs={resyncs}", flush=True)
            sim_cnt += 1
            if args.upstream_rate_limiter:
                # upstream BaseSimulator.start: sleep the rest of this step, never catch up
                elapsed = time.monotonic() - step_start
                if sim_dt - elapsed > 0:
                    time.sleep(sim_dt - elapsed)
                else:
                    late_steps += 1
            else:
                # absolute schedule: sim time tracks wall time; after a scheduling hiccup on a loaded
                # host, steps run back-to-back until the schedule is met again (bounded by max_lag_s)
                deadline = t_sched + sim_cnt * sim_dt
                now = time.monotonic()
                if now < deadline:
                    time.sleep(deadline - now)
                else:
                    late_steps += 1
                    if now - deadline > args.max_lag_s:
                        resyncs += 1
                        t_sched = now - sim_cnt * sim_dt
    finally:
        rec.stop()
        stop.set()
        sim.close()
        print(f"[sim_headless] exit after {sim_cnt} steps, falls={falls}, late_steps={late_steps}, resyncs={resyncs}", flush=True)


if __name__ == "__main__":
    main()
