"""External locations, read from environment variables (see humanoid/README.md)."""
import os
from pathlib import Path
import tempfile

PACKAGE = Path(__file__).resolve().parent
ASSETS = PACKAGE / "assets"
DEFAULT_TACD_MODEL = "weijinhuang/TACD-HY-Motion-Lite"


def _required(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} (see humanoid/README.md)")
    return value


def tacd_model():
    return os.environ.get("TACD_MODEL", DEFAULT_TACD_MODEL)


def hy_motion_root():
    """HY-Motion-1.0 source tree (hymotion package and the wooden body assets)."""
    return Path(_required("HY_MOTION_ROOT"))


def sonic_root():
    """GR00T-WholeBodyControl checkout (humanoid/sonic/install.sh)."""
    return Path(_required("SONIC_ROOT"))


def umr_root():
    return Path(_required("UMR_ROOT"))


def umr_python():
    return os.environ.get("UMR_PYTHON", "python")


def smplx_model_dir():
    value = os.environ.get("SMPLX_MODEL_DIR")
    return Path(value) if value else umr_root() / "smpl"


def isaaclab_python():
    return _required("ISAACLAB_PYTHON")


def sonic_checkpoint():
    return Path(_required("SONIC_CHECKPOINT"))


def cache_dir():
    value = os.environ.get("TACD_HUMANOID_CACHE")
    path = Path(value) if value else Path.home() / ".cache/tacd_humanoid"
    path.mkdir(parents=True, exist_ok=True)
    return path


G1_MESHES = "decoupled_wbc/sim2mujoco/resources/robots/g1/meshes"


def box_feet_xml():
    """assets/g1_boxfeet.xml with its meshdir pointing at the GR00T checkout's G1 meshes."""
    meshes = sonic_root() / G1_MESHES
    text = (ASSETS / "g1_boxfeet.xml").read_text().replace('meshdir="meshes"', f'meshdir="{meshes}"', 1)
    path = cache_dir() / "g1_boxfeet.xml"
    if not path.exists() or path.read_text() != text:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    return path


def runtime_env(run_dir, isolate_home=False):
    """Environment for this package's processes. isolate_home gives an Isaac Sim run its own
    HOME/caches so concurrent runs do not share Kit state."""
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled", TORCHDYNAMO_DISABLE="1",
               OMNI_KIT_ACCEPT_EULA="YES")
    if isolate_home:
        env.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
        # Short scratch path: Python multiprocessing puts AF_UNIX sockets (108-byte limit) in TMPDIR.
        scratch = Path(tempfile.mkdtemp(prefix="tacd_humanoid_"))
        for subdir in ["home", "cache", "torch", "tmp", "mpl"]:
            (scratch / subdir).mkdir(parents=True, exist_ok=True)
        env.update(HOME=str(scratch / "home"), XDG_CACHE_HOME=str(scratch / "cache"),
                   TORCH_EXTENSIONS_DIR=str(scratch / "torch"), TMPDIR=str(scratch / "tmp"),
                   MPLCONFIGDIR=str(scratch / "mpl"))
    pythonpath = [str(PACKAGE.parent)]
    if os.environ.get("SONIC_ROOT"):
        pythonpath.append(os.environ["SONIC_ROOT"])
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env
