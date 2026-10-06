"""SONIC side for simulation: SONIC's MuJoCo sim + SONIC's C++ deploy (sonic_v1_1, ZMQ input), kept running.

    python -m tacd_humanoid.stream.sonic_sim --run-dir runs/session --zmq-host localhost --zmq-port 5556

Brings the robot up the way the upstream sim2sim tutorial does (elastic band, ']' to start control, band
released), checks it stands on the policy, then waits until a command stream is being published on
tcp://<zmq-host>:<zmq-port> and only then switches the deploy to ZMQ streaming mode (Enter), because
the deploy must not sit in ZMQ mode without data.  Runs until SIGINT/SIGTERM or until --stop-file exists,
then stops recording, exits the deploy ('O') and quits the sim.  The simulated robot state is recorded
from start to stop in <run-dir>/session.npz (50 Hz, CLOCK_MONOTONIC wall times).
"""
import argparse
import json
from pathlib import Path
import signal
import time

from ..paths import PACKAGE
from .service import BASE_ENV, CLONE, DEPLOY_BIN, DEPLOY_DIR, DEPLOY_LIBRARY_PATH, SIM_PY, Service, log, pick_gpu


def wait_for_stream(host, port, topic, timeout):
    import zmq
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    try:
        if not sock.poll(int(timeout * 1000)):
            return False
        sock.recv()
        return True
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", type=int, default=None, help="GPU for the deploy's TensorRT engines")
    parser.add_argument("--zmq-host", default="localhost", help="host of our command server")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--zmq-topic", default="pose")
    parser.add_argument("--stream-timeout-s", type=float, default=3600)
    parser.add_argument("--stop-file", default=None)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--attempts", type=int, default=3, help="restarts if the robot does not stand after release")
    parser.add_argument("--init-timeout", type=float, default=900, help="first start builds TensorRT engines")
    args = parser.parse_args()
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    registry = out / "processes.jsonl"
    gpu = args.gpu if args.gpu is not None else pick_gpu()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    for attempt in range(1, args.attempts + 1):
        sim = Service("sim", out, CLONE, {**BASE_ENV, "PYTHONPATH": f"{CLONE}:{PACKAGE.parent}"},
                      [str(SIM_PY), "-u", "-m", "tacd_humanoid.stream.sonic_sim_headless", "--out-dir", str(out),
                       "--record-from-start", f"session_attempt{attempt}"], registry)
        deploy = None
        try:
            sim.wait_for("sim_t=", 120)
            deploy = Service("deploy", out, DEPLOY_DIR, {**BASE_ENV, "LD_LIBRARY_PATH": DEPLOY_LIBRARY_PATH,
                                                         "CUDA_VISIBLE_DEVICES": str(gpu)},
                             [str(DEPLOY_BIN), "lo", "policy/sonic_v1_1/model_decoder.onnx", "reference/example/",
                              "--obs-config", "policy/sonic_v1_1/observation_config.yaml",
                              "--encoder-file", "policy/sonic_v1_1/model_encoder.onnx",
                              "--input-type", "zmq", "--zmq-host", args.zmq_host, "--zmq-port", str(args.zmq_port),
                              "--zmq-topic", args.zmq_topic, "--disable-crc-check",
                              "--target-motion-logfile", str(out / "target_motion.csv"),
                              "--enable-csv-logs", "--logs-dir", str(out / "deploy_logs")], registry)
            deploy.wait_for("Init Done", args.init_timeout)
            for _ in range(2):            # lower the band 0.2 m so the feet touch the ground
                sim.send("7\n"); time.sleep(0.2)
            time.sleep(1.0)
            mark = deploy.log_size()
            deploy.send("]")
            deploy.wait_for("transitioning to CONTROL state", 20, mark)
            time.sleep(1.0)
            before = sim.query_status("prerelease")
            sim.send("9\n")
            sim.wait_for("ElasticBand enable: False", 10)
            time.sleep(args.settle_s)
            after = sim.query_status("settled")
            falls, z = after["falls"] - before["falls"], after["pelvis_z"]
            log(f"attempt {attempt}: standing check falls={falls:.0f} pelvis_z={z:.3f}")
            if falls > 0 or z < 0.7:
                raise RuntimeError("robot did not stand after release")
            log(f"standing; waiting for a command stream on tcp://{args.zmq_host}:{args.zmq_port}")
            if not wait_for_stream(args.zmq_host, args.zmq_port, args.zmq_topic, args.stream_timeout_s):
                raise TimeoutError("no command stream")
            mark = deploy.log_size()
            deploy.send("\n")
            deploy.wait_for("ZMQ STREAMING MODE: ENABLED", 10, mark)
            log("ZMQ streaming mode enabled")
            (out / "SONIC_READY").write_text(json.dumps({"attempt": attempt, "time": time.time()}))
            while not stop["flag"] and not (args.stop_file and Path(args.stop_file).exists()):
                if not (sim.alive() and deploy.alive()):
                    raise RuntimeError("sim or deploy exited")
                time.sleep(0.2)
            sim.send("rec_stop\n")
            sim.wait_for(f"-> {out / f'session_attempt{attempt}'}.npz", 30)
            (out / "session.npz").unlink(missing_ok=True)
            (out / "session.npz").symlink_to(f"session_attempt{attempt}.npz")
            log("stopped on request")
            return
        except RuntimeError as e:
            log(f"attempt {attempt} failed: {e}")
            if "did not stand" not in str(e) or attempt == args.attempts:
                raise
        finally:
            if deploy is not None:
                deploy.stop("O", expect_argv0=str(DEPLOY_BIN))
            sim.stop("quit\n", expect_argv0=str(SIM_PY))


if __name__ == "__main__":
    main()
