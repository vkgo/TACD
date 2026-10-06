"""Run SONIC's MuJoCo sim and C++ deploy as detached services with a stdin FIFO each.

Each service is started with setsid/nohup through detach.sh in an empty environment (env -i), with its
own log, PID file and FIFO; every start/stop is appended to <run-dir>/processes.jsonl.
Locations come from environment variables:
  SONIC_ROOT              GR00T-WholeBodyControl checkout (humanoid/sonic/install.sh)
  SONIC_SIM_PYTHON        python of the SONIC sim environment
  SONIC_DEPLOY_BIN        g1_deploy_onnx_ref (default: $SONIC_ROOT/gear_sonic_deploy/target/release/)
  SONIC_DEPLOY_LIBRARY_PATH  LD_LIBRARY_PATH for the deploy (TensorRT, ONNX Runtime, CUDA runtime);
                          printed by humanoid/sonic/build_deploy.sh
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ..paths import sonic_root

CLONE = sonic_root()
DEPLOY_DIR = CLONE / "gear_sonic_deploy"
DEPLOY_BIN = Path(os.environ.get("SONIC_DEPLOY_BIN", str(DEPLOY_DIR / "target/release/g1_deploy_onnx_ref")))
SIM_PY = Path(os.environ.get("SONIC_SIM_PYTHON", "python"))
DEPLOY_LIBRARY_PATH = os.environ.get("SONIC_DEPLOY_LIBRARY_PATH", "")
BASE_ENV = {"HOME": os.environ.get("HOME", "/tmp"), "USER": os.environ.get("USER", ""),
            "LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def pick_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    free = [int(i) for i, m in (l.split(",") for l in out.strip().splitlines()) if int(m) < 500]
    if not free:
        raise SystemExit("no GPU with memory.used < 500 MiB")
    return free[-1]


class Service:
    def __init__(self, name, workdir: Path, cwd: Path, env: dict, argv: list, registry: Path):
        self.name, self.workdir, self.registry = name, workdir, registry
        self.pidfile, self.logfile, self.fifo = (workdir / f"{name}.pid", workdir / f"{name}.log",
                                                 workdir / f"{name}.fifo")
        for p in (self.pidfile, self.fifo):
            if p.exists():
                p.unlink()
        os.mkfifo(self.fifo)
        cmd = ["setsid", "nohup", str(Path(__file__).with_name("detach.sh")), str(self.pidfile), str(self.logfile),
               str(self.fifo), "--", "env", "-i", *[f"{k}={v}" for k, v in env.items()], *argv]
        subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + 10
        while not self.pidfile.exists() and time.time() < deadline:
            time.sleep(0.05)
        self.pid = int(self.pidfile.read_text())
        time.sleep(0.3)
        self.argv0 = self.read_argv0()
        self.record(event="started", argv=argv, cwd=str(cwd))
        log(f"{name}: pid {self.pid} argv0 {self.argv0}")

    def read_argv0(self):
        try:
            return Path(f"/proc/{self.pid}/cmdline").read_bytes().split(b"\0")[0].decode()
        except FileNotFoundError:
            return None

    def record(self, **kw):
        with open(self.registry, "a") as f:
            f.write(json.dumps({"service": self.name, "pid": self.pid, "time": now(), **kw}) + "\n")

    def alive(self):
        return self.read_argv0() is not None

    def send(self, text):
        fd = os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, text.encode())
        finally:
            os.close(fd)

    def wait_for(self, pattern, timeout, start_offset=0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self.logfile.read_text(errors="replace") if self.logfile.exists() else ""
            if pattern in text[start_offset:]:
                return True
            if not self.alive():
                raise RuntimeError(f"{self.name} died while waiting for {pattern!r}")
            time.sleep(0.1)
        raise TimeoutError(f"{self.name}: {pattern!r} not seen in {timeout}s")

    def query_status(self, tag):
        """Ask the headless sim for its counters; returns dict of the STATUS line."""
        mark = self.log_size()
        self.send(f"status {tag}\n")
        self.wait_for(f"STATUS id={tag} ", 10, mark)
        line = [l for l in self.logfile.read_text(errors="replace")[mark:].splitlines() if f"STATUS id={tag} " in l][-1]
        return {k: float(v) for k, v in (kv.split("=") for kv in line.split()[3:])}

    def log_size(self):
        return len(self.logfile.read_text(errors="replace")) if self.logfile.exists() else 0

    def stop(self, graceful_text=None, expect_argv0=None, timeout=15):
        how = "already-exited"
        if self.alive() and graceful_text is not None:
            self.send(graceful_text)
            deadline = time.time() + timeout
            while self.alive() and time.time() < deadline:
                time.sleep(0.2)
            how = "graceful" if not self.alive() else how
        if self.alive():
            argv0 = self.read_argv0()
            if expect_argv0 and argv0 and Path(argv0).name != Path(expect_argv0).name:
                raise RuntimeError(f"refusing to signal pid {self.pid}: argv0 {argv0} != {expect_argv0}")
            os.kill(self.pid, signal.SIGTERM)
            deadline = time.time() + timeout
            while self.alive() and time.time() < deadline:
                time.sleep(0.2)
            how = "SIGTERM"
            if self.alive():
                os.kill(self.pid, signal.SIGKILL)
                time.sleep(0.5)
                how = "SIGKILL"
        self.record(event="stopped", how=how, alive_after=self.alive())
        log(f"{self.name}: stopped ({how}), alive={self.alive()}")
        return how
