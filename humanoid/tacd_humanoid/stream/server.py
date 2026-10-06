"""Resident text-to-humanoid server: TACD + UMR (umr_tuned_f0) stay loaded; type a prompt, press Enter to send.

    CUDA_VISIBLE_DEVICES=0 python -m tacd_humanoid.stream.server --bind tcp://0.0.0.0:5556 --out runs/server_session

The process binds a ZMQ PUB socket (SONIC Protocol v1, topic "pose") and publishes a 50 Hz stream from
the moment it starts.  With nothing queued the stream is the SONIC default standing pose.  A prompt is
generated with TACD (8 steps), retargeted with UMR (umr_tuned_f0), given the grounded stand-in prefix
(v2g) and stand-out suffix (s2g), and kept ready.  Pressing Enter splices it into the stream: the clip is
moved (yaw + xy) so that its first standing frame coincides with the standing frame it replaces, so the
deploy sees one continuous motion with monotonically increasing frame indices and never resets.  While a
clip is playing, a newly sent clip starts when the current one is back in the standing pose.

Console:
    <text>          generate and retarget (replaces the ready clip)
    <Enter>         send the ready clip
    :d <seconds>    duration of the next generation (default 4)
    :s <seed>       seed of the next generation (default 0; incremented after every generation)
    :wait           block until the stream is standing idle again
    :q              quit (the stream keeps standing for --quit-hold-s first)
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from . import protocol


def _yaw(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def place_at(qpos50, anchor):
    """Rigidly move a clip in the horizontal plane so that frame 0 has the anchor's root xy and yaw."""
    q = np.array(qpos50, np.float64)
    dyaw = _yaw(anchor[3:7]) - _yaw(q[0, 3:7])
    rot = Rotation.from_euler("z", dyaw)
    xy = q[:, :2] - q[0, :2]
    q[:, :2] = rot.apply(np.c_[xy, np.zeros(len(q))])[:, :2] + anchor[:2]
    q[:, 3:7] = (rot * Rotation.from_quat(q[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
    return q


class StreamBuffer:
    """Global 50 Hz frame timeline.  Frames past the last spliced clip repeat the last frame."""

    def __init__(self, standing, length):
        self.frames = [np.asarray(standing, np.float64)]  # frames[i] is global frame base + i
        self.base, self.length, self.content_end = 0, length, 0
        self.splices = []

    def _extend_to(self, end):
        need = end - (self.base + len(self.frames))
        if need > 0:
            self.frames.extend([self.frames[-1]] * need)

    def window(self, k):
        """Frames [k, k+L+1) (one extra frame for the forward-difference velocity of frame k+L-1)."""
        self._extend_to(k + self.length + 1)
        drop = k - self.base - 10  # keep a few played frames, drop older ones
        if drop > 0:
            del self.frames[:drop]
            self.base += drop
        i = k - self.base
        return np.asarray(self.frames[i:i + self.length + 1])

    def splice(self, qpos50, k, margin=2):
        """Place a clip after the current one (or right away when standing idle)."""
        start = max(k + margin, self.content_end)
        self._extend_to(start)
        anchor = self.frames[start - 1 - self.base]
        clip = place_at(qpos50, anchor)
        if np.dot(clip[0, 3:7], anchor[3:7]) < 0:  # same quaternion hemisphere as the frame it follows
            clip[:, 3:7] *= -1
        jump = float(np.abs(clip[0] - anchor).max())
        del self.frames[start - self.base:]
        self.frames.extend(clip)
        self.content_end = start + len(clip)
        self.splices.append(dict(start=start, end=self.content_end, junction_max_abs=jump, sent_at_k=k))
        return start, self.content_end, jump


def _publisher(conn, bind, topic, length, standing, log_path):
    """Publisher process.  Frame k is due at t0 + k/50 (wall clock); if the process is ever late it skips
    to the frame that is due instead of sending a burst, so the deploy's playback stays on our clock."""
    import zmq
    buffer = StreamBuffer(standing, length)
    perm = protocol.mujoco_to_isaaclab()
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.setsockopt(zmq.LINGER, 1000)
    sock.bind(bind)
    log_k, log_t, log_q = [], [], []
    skipped, late, k = 0, 0, -1
    t0 = time.monotonic()
    conn.send(("started", t0))
    while True:
        while conn.poll(0):
            cmd = conn.recv()
            if cmd[0] == "splice":
                conn.send(buffer.splice(cmd[1], k + 1))
            elif cmd[0] == "status":
                conn.send((k + 1, buffer.content_end))
            elif cmd[0] == "stop":
                sock.close()
                np.savez(log_path, frame_index=np.array(log_k), send_time=np.array(log_t), qpos=np.array(log_q))
                conn.send(dict(messages=len(log_k), skipped_frames=skipped, late_messages=late, splices=buffer.splices))
                return
        now = time.monotonic()
        due = int((now - t0) * protocol.CONTROL_FPS)
        if due <= k:
            time.sleep(max(0.0, t0 + (k + 1) / protocol.CONTROL_FPS - now))
            continue
        skipped += due - k - 1
        late += now - (t0 + due / protocol.CONTROL_FPS) > 0.01
        k = due
        window = buffer.window(k)
        joints = window[:, 7:][:, perm]
        quat = window[:, 3:7] / np.linalg.norm(window[:, 3:7], axis=1, keepdims=True)
        fields = dict(joint_pos=joints[:-1].astype(np.float32),
                      joint_vel=((joints[1:] - joints[:-1]) * protocol.CONTROL_FPS).astype(np.float32),
                      body_quat=quat[:-1].astype(np.float32),
                      frame_index=np.arange(k, k + length, dtype=np.int64),
                      catch_up=np.array([True]), timestamp_monotonic=np.array([time.monotonic()]))
        sock.send(protocol.pack_message(fields, topic=topic))
        log_k.append(k); log_t.append(time.monotonic()); log_q.append(window[0])


class Stream:
    """Handle to the publisher process (its own process, so model loading and retargeting in this
    process can never delay a message)."""

    def __init__(self, bind, topic, lookahead, standing, log_path):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_publisher, args=(child, bind, topic, lookahead, standing, str(log_path)),
                                   daemon=True)
        self.process.start()
        self.conn.recv()

    def splice(self, qpos50):
        self.conn.send(("splice", qpos50))
        return self.conn.recv()

    def status(self):
        self.conn.send(("status",))
        return self.conn.recv()

    def idle(self):
        k, end = self.status()
        return k >= end

    def close(self):
        self.conn.send(("stop",))
        summary = self.conn.recv()
        self.process.join(timeout=5)
        return summary


class Pipeline:
    """TACD generator and UMR retarget worker, both resident."""

    def __init__(self, device="cuda:0", retargeter="umr_tuned_f0", suffix="s2g"):
        from ..generate import MotionGenerator
        from ..retarget import make_retargeter
        self.suffix = suffix
        t = time.perf_counter()
        self.generator = MotionGenerator(device)
        print(f"[server] TACD loaded in {time.perf_counter() - t:.1f} s", flush=True)
        t = time.perf_counter()
        self.retargeter = make_retargeter(retargeter)
        self.retargeter_name = retargeter
        print(f"[server] {retargeter} worker ready in {time.perf_counter() - t:.1f} s", flush=True)

    def __call__(self, text, duration=4.0, seed=0, out=None):
        from ..export import canonical_qpos
        from ..transition import prepend_stand_transition, append_stand_suffix, DEFAULT_STAND_VARIANT
        times = {}
        t = time.perf_counter()
        human = self.generator.generate(text, duration, seed)
        times["generate"] = time.perf_counter() - t
        t = time.perf_counter()
        robot = self.retargeter.retarget(human)
        times["retarget"] = time.perf_counter() - t
        t = time.perf_counter()
        reference = prepend_stand_transition(robot, variant=DEFAULT_STAND_VARIANT)
        reference = append_stand_suffix(reference, variant=self.suffix)
        qpos50 = canonical_qpos(reference, protocol.CONTROL_FPS)
        times["transitions"] = time.perf_counter() - t
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            human.save(out / "human.npz")
            robot.save(out / f"{self.retargeter_name}.npz")
            reference.save(out / "reference_robot.npz")
            np.save(out / "qpos50.npy", qpos50)
            meta = reference.metadata
            (out / "clip.json").write_text(json.dumps(dict(
                text=text, duration=duration, seed=seed, retargeter=self.retargeter_name,
                stand_prefix_seconds=meta.get("stand_prefix_seconds"), generated_end_seconds=meta.get("generated_end_seconds"),
                stand_suffix_seconds=meta.get("stand_suffix_seconds"), frames_50hz=len(qpos50), seconds=times), indent=2))
        return qpos50, reference.metadata, times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bind", default="tcp://0.0.0.0:5556")
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--lookahead", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--retargeter", default="umr_tuned_f0", choices=["umr_tuned_f0"])
    parser.add_argument("--out", required=True, help="session directory (one subdirectory per generated clip)")
    parser.add_argument("--warmup", default="a person stands still", help="prompt run once at start-up ('' to skip)")
    parser.add_argument("--quit-hold-s", type=float, default=2.0)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    from ..paths import runtime_env
    os.environ.update(runtime_env(out))
    os.environ.setdefault("MUJOCO_GL", "egl")
    from ..transition import standing_pose

    stream = Stream(args.bind, args.topic, args.lookahead, standing_pose(), out / "stream_log.npz")
    print(f"[server] streaming the standing pose on {args.bind} (topic {args.topic}, {args.lookahead}-frame lookahead)",
          flush=True)
    pipeline = Pipeline(args.device, args.retargeter)
    if args.warmup:
        t = time.perf_counter()
        pipeline(args.warmup, 2.0, 0)
        print(f"[server] warm-up clip done in {time.perf_counter() - t:.1f} s", flush=True)

    duration, seed, ready, count = 4.0, 0, None, 0
    print("[server] ready. Type a prompt; Enter sends the generated clip; :q quits.", flush=True)
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.startswith(":d "):
            duration = float(line.split()[1]); print(f"[server] duration {duration} s", flush=True)
        elif line.startswith(":s "):
            seed = int(line.split()[1]); print(f"[server] seed {seed}", flush=True)
        elif line == ":wait":
            while not stream.idle():
                time.sleep(0.05)
            print("[server] standing idle", flush=True)
        elif line.startswith(":waitfile "):  # for scripted sessions: block until a file exists
            while not Path(line.split(None, 1)[1]).exists():
                time.sleep(0.2)
        elif line == ":q":
            break
        elif line.strip() == "":
            if ready is None:
                print("[server] nothing to send; type a prompt first", flush=True)
                continue
            start, end, jump = stream.splice(ready["qpos50"])
            now = stream.status()[0]
            ready["sent"] = dict(start_frame=start, end_frame=end, junction_max_abs=jump)
            (ready["dir"] / "sent.json").write_text(json.dumps(ready["sent"]))
            print(f"[server] sent '{ready['text']}': starts in {(start - now) / 50:.2f} s, "
                  f"standing again in {(end - now) / 50:.2f} s", flush=True)
            ready = None
        else:
            count += 1
            clip_dir = out / f"{count:03d}"
            t = time.perf_counter()
            try:
                qpos50, meta, times = pipeline(line, duration, seed, clip_dir)
            except Exception as e:  # keep serving; the stream is unaffected
                print(f"[server] generation failed: {type(e).__name__}: {e}", flush=True)
                continue
            ready = dict(text=line, qpos50=qpos50, dir=clip_dir)
            print(f"[server] ready '{line}' (seed {seed}, {duration:.1f} s): TACD {times['generate']:.2f} s, "
                  f"UMR {times['retarget']:.1f} s, total {time.perf_counter() - t:.1f} s; "
                  f"clip {len(qpos50) / 50:.1f} s incl. stand-in/out. Press Enter to send.", flush=True)
            seed += 1

    while not stream.idle():
        time.sleep(0.05)
    time.sleep(args.quit_hold_s)
    summary = stream.close()
    (out / "session.json").write_text(json.dumps(dict(bind=args.bind, **summary), indent=2))
    if hasattr(pipeline.retargeter, "close"):
        pipeline.retargeter.close()
    print(f"[server] stopped: {summary['messages']} messages, {summary['skipped_frames']} skipped frames, "
          f"log {out / 'stream_log.npz'}", flush=True)


if __name__ == "__main__":
    main()
