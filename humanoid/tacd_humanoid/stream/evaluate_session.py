"""Score a server session (tacd_humanoid.stream.server) against SONIC's simulated execution (tacd_humanoid.stream.sonic_sim).

Every sent clip is compared on its generated segment (stand-in prefix and stand-out suffix excluded),
with the robot state recorded at wall time w compared with the stream frame that was due at w
(frame f is due when message f is published).  Joint metrics use tacd_humanoid.tracking_metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..motion import RobotMotion
from ..tracking_metrics import generated_tracking_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server-dir", required=True)
    parser.add_argument("--sonic-dir", required=True)
    args = parser.parse_args()
    server, sonic = Path(args.server_dir), Path(args.sonic_dir)
    with np.load(server / "stream_log.npz") as d:
        frames, send_time = d["frame_index"], d["send_time"]
    with np.load(sonic / "session.npz") as d:
        sim = {k: d[k] for k in d.files}
    wall, q_robot = sim["wall_time"], sim["qpos"].astype(np.float64)
    rows = {}
    for clip_dir in sorted(p for p in server.iterdir() if (p / "sent.json").exists()):
        sent = json.loads((clip_dir / "sent.json").read_text())
        meta = json.loads((clip_dir / "clip.json").read_text())
        reference = RobotMotion.load(clip_dir / "reference_robot.npz")
        t0 = float(np.interp(sent["start_frame"], frames, send_time))
        t1 = float(np.interp(sent["end_frame"], frames, send_time))
        keep = (wall >= t0) & (wall <= t1)
        q, t = q_robot[keep], wall[keep] - t0
        z = q[:, 2]
        row = dict(text=meta["text"], seed=meta["seed"], start_frame=sent["start_frame"],
                   junction_max_abs=sent["junction_max_abs"], states=int(keep.sum()),
                   fall=bool(sim["fall_flag"][keep].any() or (z < 0.5).any()), min_pelvis_z_m=float(z.min()),
                   final_pelvis_z_m=float(z[-1]), latency_s=meta["seconds"])
        row.update(generated_tracking_metrics(q, reference, 50.0, t))
        row.pop("per_joint_rmse_rad", None)
        # diagnostic: delay of the robot behind the stream that minimises the generated-segment joint RMSE
        lags = np.round(np.arange(-0.5, 2.51, 0.02), 2)
        rmse = [generated_tracking_metrics(q, reference, 50.0, t - lag).get("joint_rmse_rad", np.inf) for lag in lags]
        row.update(best_lag_s=float(lags[int(np.argmin(rmse))]), joint_rmse_at_best_lag=float(np.min(rmse)))
        rows[clip_dir.name] = row
    released = sim["band_enabled"] < 0.5
    first = min((float(np.interp(json.loads((p / "sent.json").read_text())["start_frame"], frames, send_time))
                 for p in server.iterdir() if (p / "sent.json").exists()), default=wall[-1])
    idle = released & (wall < first) & (wall >= send_time[0])
    summary = dict(clips=rows, standing_before_first_clip_s=float(idle.sum() / 50),
                   fall_while_standing=bool(sim["fall_flag"][idle].any() or (q_robot[idle, 2] < 0.5).any()),
                   fall_anywhere_after_release=bool(sim["fall_flag"][released].any() or (q_robot[released, 2] < 0.5).any()))
    (server / "evaluation.json").write_text(json.dumps(summary, indent=2))
    for name, r in rows.items():
        print(f"{name} {r['text'][:40]:40s} fall={r['fall']} minz={r['min_pelvis_z_m']:.3f} "
              f"joint={r.get('joint_rmse_rad', float('nan')):.4f} legs={r.get('legs_rmse_rad', float('nan')):.4f} "
              f"arms={r.get('arms_rmse_rad', float('nan')):.4f} lag={r['best_lag_s']:.2f}s@{r['joint_rmse_at_best_lag']:.4f} junction={r['junction_max_abs']:.2e} "
              f"TACD={r['latency_s']['generate']:.2f}s UMR={r['latency_s']['retarget']:.1f}s")
    print(json.dumps({k: v for k, v in summary.items() if k != "clips"}))


if __name__ == "__main__":
    main()
