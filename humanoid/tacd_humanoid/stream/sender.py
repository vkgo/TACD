"""Our command process: stream a RobotMotion to SONIC's deploy over ZMQ Protocol v1.

    python -m tacd_humanoid.stream.sender --reference <clip>/reference_robot.npz --log out/<clip>_sent.npz

The deploy (``g1_deploy_onnx_ref --input-type zmq --zmq-host <us> --zmq-port 5556 --zmq-topic pose``)
connects a SUB socket to us, so this process binds the PUB end.  Every 1/50 s it publishes frames
[k, k+L) of the 50 Hz stream (global frame_index k..k+L-1); the deploy's StreamedMotionMerger keeps
a sliding window over these and plays one frame per control tick.  The window start must strictly
increase and the window end must strictly grow, otherwise the merger forces a catch-up reset, so
we stop publishing once the window reaches the last frame; the stream therefore ends with a standing
hold longer than L frames so that every non-hold frame is still delivered live.

Stream content: the reference already starts with the grounded stand-in prefix (v2g); we append the
mirrored stand-out suffix (tacd_humanoid.transition.append_stand_suffix, s2g by default) and an extra standing
hold, then resample to 50 Hz on the exact source grid (tacd_humanoid.export.canonical_qpos).
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from ..motion import RobotMotion
from ..export import canonical_qpos
from ..transition import append_stand_suffix
from . import protocol


def build_stream(reference: RobotMotion, suffix="s2g", final_hold_s=2.5, suffix_hold_s=1.0):
    """Returns (motion_30fps_with_suffix_and_hold, qpos50, info)."""
    motion = append_stand_suffix(reference, variant=suffix, suffix_hold_s=suffix_hold_s)
    hold = int(round(final_hold_s * motion.fps))
    if hold > 0:
        q = np.concatenate([motion.qpos, np.repeat(motion.qpos[-1:], hold, axis=0)])
        motion = RobotMotion(q, motion.fps, {**motion.metadata, "stream_final_hold_frames": hold,
                                             "stream_final_hold_seconds": hold / motion.fps},
                             motion.source_qpos)
        motion.validate()
    qpos50 = canonical_qpos(motion, protocol.CONTROL_FPS)
    meta = motion.metadata
    info = dict(frames_50hz=len(qpos50), seconds=len(qpos50) / protocol.CONTROL_FPS,
                stand_prefix_seconds=float(meta.get("stand_prefix_seconds", 0.0)),
                generated_end_seconds=float(meta["generated_end_seconds"]),
                stand_suffix_variant=suffix, stand_suffix_seconds=float(meta.get("stand_suffix_seconds", 0.0)),
                stream_final_hold_seconds=float(meta.get("stream_final_hold_seconds", 0.0)))
    return motion, qpos50, info


def _sleep_until(deadline):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.002) if remaining < 0.004 else remaining - 0.002)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, help="RobotMotion npz (30 fps, MuJoCo order)")
    parser.add_argument("--log", required=True, help="npz written after streaming (sent arrays + send times)")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5556")
    parser.add_argument("--topic", default="pose")
    parser.add_argument("--lookahead", type=int, default=100, help="frames per message (L)")
    parser.add_argument("--suffix", default="s2g", choices=["none", "s2g"])
    parser.add_argument("--final-hold-s", type=float, default=None,
                        help="extra standing hold after the suffix (default: L/50 + 0.5 s)")
    parser.add_argument("--connect-wait-s", type=float, default=1.0,
                        help="pause after bind so the deploy's SUB connection/subscription is in place")
    parser.add_argument("--preroll-s", type=float, default=0.0,
                        help="stream the first frame as a standing hold for this long before the clip")
    parser.add_argument("--ready-file", default=None, help="touched after the first message is sent")
    parser.add_argument("--no-catch-up", action="store_true", help="send catch_up=false")
    args = parser.parse_args()
    import zmq

    reference = RobotMotion.load(args.reference)
    final_hold = args.final_hold_s if args.final_hold_s is not None else args.lookahead / protocol.CONTROL_FPS + 0.5
    motion, qpos50, info = build_stream(reference, suffix=args.suffix, final_hold_s=final_hold)
    # Pre-roll: hold the clip's first (standing) frame so the deploy already receives a live stream when
    # the operator switches it to ZMQ mode.  SONIC's deploy must not sit in ZMQ mode without data: in
    # our first runs the robot fell ~1.3 s after ENTER while our sender was still starting up.
    preroll = int(round(args.preroll_s * protocol.CONTROL_FPS))
    stream_q = np.concatenate([np.repeat(qpos50[:1], preroll, axis=0), qpos50]) if preroll else qpos50
    arrays = protocol.stream_arrays(stream_q)
    n, length = len(stream_q), args.lookahead
    if n < length + 1:
        raise SystemExit("stream shorter than one lookahead window")
    # self-check: the first message decodes back to the exact arrays we meant to send
    check = protocol.unpack_message(protocol.window_message(arrays, 0, length, timestamp=0.0), args.topic)
    for key in ("joint_pos", "joint_vel", "body_quat", "frame_index"):
        np.testing.assert_array_equal(check[key], arrays[key][:length])

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.setsockopt(zmq.LINGER, 1000)
    sock.bind(args.bind)
    print(f"[sender] bound {args.bind}; {n} frames ({n / 50:.2f} s, preroll {preroll}), L={length}, "
          f"prefix {info['stand_prefix_seconds']:.2f}s, generated_end {info['generated_end_seconds']:.2f}s, "
          f"suffix {args.suffix} {info['stand_suffix_seconds']:.2f}s, hold {info['stream_final_hold_seconds']:.2f}s",
          flush=True)
    time.sleep(args.connect_wait_s)

    count = n - length + 1  # last message covers [n-L, n-1]
    send_times = np.zeros(count)
    late = 0
    t0 = time.monotonic()
    for k in range(count):
        deadline = t0 + k / protocol.CONTROL_FPS
        if time.monotonic() > deadline + 0.01:
            late += 1
        _sleep_until(deadline)
        now = time.monotonic()
        sock.send(protocol.window_message(arrays, k, length, catch_up=not args.no_catch_up, timestamp=now,
                                          topic=args.topic))
        send_times[k] = now
        if k == 0 and args.ready_file:
            Path(args.ready_file).write_text(f"{now}\n")
        if k % 250 == 0:
            print(f"[sender] k={k}/{count} t={now - t0:.2f}s", flush=True)
    print(f"[sender] done: {count} messages in {send_times[-1] - t0:.2f}s, late(>10ms)={late}", flush=True)
    time.sleep(0.5)
    sock.close()

    out = Path(args.log); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, qpos50=qpos50, send_times=send_times[preroll:], preroll_send_times=send_times[:preroll],
             preroll_frames=preroll, lookahead=length, fps=protocol.CONTROL_FPS,
             motion_qpos30=motion.qpos, motion_metadata=json.dumps(motion.metadata),
             motion_source_qpos=(motion.source_qpos if motion.source_qpos is not None else np.zeros((0, 36))),
             info=json.dumps({**info, "late_messages": late, "messages": count, "bind": args.bind,
                              "reference": str(Path(args.reference).resolve())}),
             **{f"stream_{k}": v[preroll:] for k, v in arrays.items()})
    with np.load(tmp) as chk:
        assert chk["send_times"].shape == (count - preroll,)
    os.replace(tmp, out)
    print(f"[sender] log -> {out}", flush=True)


if __name__ == "__main__":
    main()
