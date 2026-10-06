#!/usr/bin/env bash
# Install SONIC (GR00T-WholeBodyControl + GEAR-SONIC sonic_v1_1 models) at the versions in pins.json.
#
#   bash humanoid/sonic/install.sh <prefix> [--isaaclab]
#
# <prefix>/GR00T-WholeBodyControl   checkout at the pinned commit, with Git LFS assets (G1 meshes)
#   gear_sonic_deploy/policy/sonic_v1_1/{model_encoder,model_decoder}.onnx, observation_config.yaml
#   gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
# <prefix>/sonic_v1_1/{last.pt,config.yaml,model_config.yaml}   with --isaaclab (IsaacLab path)
#
# Every downloaded file is checked against its sha256 in pins.json.  Needs git, git-lfs and a python
# with huggingface_hub (PYTHON, default python3).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: $0 <prefix> [--isaaclab]" >&2; exit 2; }
PREFIX="$(mkdir -p "$1" && cd "$1" && pwd)"
ISAACLAB=0
[ "${2:-}" = "--isaaclab" ] && ISAACLAB=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
pin() { "$PYTHON" -c "import json,sys; d=json.load(open('$HERE/pins.json')); print(eval('d'+sys.argv[1]))" "$1"; }

REPO="$(pin "['gr00t_wholebodycontrol']['repo']")"
COMMIT="$(pin "['gr00t_wholebodycontrol']['commit']")"
CLONE="$PREFIX/GR00T-WholeBodyControl"
if [ ! -d "$CLONE/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO" "$CLONE"
fi
git -C "$CLONE" fetch --quiet origin "$COMMIT" || true
git -C "$CLONE" -c advice.detachedHead=false checkout "$COMMIT"
git -C "$CLONE" lfs install --local
git -C "$CLONE" lfs pull
test "$(git -C "$CLONE" rev-parse HEAD)" = "$COMMIT"

"$PYTHON" - "$HERE/pins.json" "$PREFIX" "$ISAACLAB" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

pins, prefix, isaaclab = json.load(open(sys.argv[1]))["gear_sonic_hf"], Path(sys.argv[2]), sys.argv[3] == "1"
jobs = [(name, spec["sha256"], prefix / "GR00T-WholeBodyControl" / spec["dest"])
        for name, spec in pins["deploy_files"].items()]
if isaaclab:
    jobs += [(name, spec["sha256"], prefix / name) for name, spec in pins["isaaclab_files"].items()]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

for name, digest, dest in jobs:
    if not (dest.exists() and sha256(dest) == digest):
        source = hf_hub_download(pins["repo"], name, revision=pins["revision"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    if sha256(dest) != digest:
        sys.exit(f"sha256 mismatch: {dest}")
    print(f"ok {dest}")
PY

echo
echo "export SONIC_ROOT=$CLONE"
[ "$ISAACLAB" = 1 ] && echo "export SONIC_CHECKPOINT=$PREFIX/sonic_v1_1/last.pt"
echo "Next: bash $HERE/build_deploy.sh $PREFIX"
