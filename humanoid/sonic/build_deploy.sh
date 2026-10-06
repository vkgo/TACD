#!/usr/bin/env bash
# Build SONIC's C++ deploy (g1_deploy_onnx_ref) without root and without a system CUDA/TensorRT.
#
#   bash humanoid/sonic/build_deploy.sh <prefix>        (after install.sh <prefix>)
#
# Compilers, the CUDA 12.8 runtime and zmq/msgpack/eigen come from a conda env built from
# humanoid/envs/sonic_deploy.conda.txt (created at <prefix>/env unless SONIC_DEPLOY_ENV is set);
# TensorRT 10.13.3.9 and ONNX Runtime 1.16.3 are the official x86_64 tarballs and msgpack's
# Boost.Predef headers come from conda-forge libboost-headers 1.90.0, all unpacked under
# <prefix>/thirdparty.  TensorRT is downloaded from NVIDIA's developer site, which needs a login: if
# the download fails, put the archive named in pins.json into <prefix>/downloads and run again.
# Same build as upstream `just build`, from an empty environment so nothing from the caller leaks in.
set -euo pipefail
if [ -z "${_SONIC_CLEAN_ENV:-}" ]; then
  exec env -i _SONIC_CLEAN_ENV=1 HOME="$HOME" USER="${USER:-}" LANG=C.UTF-8 PATH=/usr/bin:/bin \
    JOBS="${JOBS:-8}" CONDA_EXE="${CONDA_EXE:-conda}" SONIC_DEPLOY_ENV="${SONIC_DEPLOY_ENV:-}" \
    http_proxy="${http_proxy:-}" https_proxy="${https_proxy:-}" PYTHON="${PYTHON:-python3}" \
    bash "$0" "$@"
fi
[ $# -ge 1 ] || { echo "usage: $0 <prefix>" >&2; exit 2; }
PREFIX="$(cd "$1" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY="$PREFIX/GR00T-WholeBodyControl/gear_sonic_deploy"
ENV_PREFIX="${SONIC_DEPLOY_ENV:-$PREFIX/env}"
THIRD="$PREFIX/thirdparty"
DOWNLOADS="$PREFIX/downloads"
JOBS="${JOBS:-8}"
mkdir -p "$THIRD" "$DOWNLOADS"
pin() { "$PYTHON" -c "import json,sys; d=json.load(open('$HERE/pins.json'))['deploy_build']; print(eval('d'+sys.argv[1]))" "$1"; }

if [ ! -x "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++" ]; then
  "$CONDA_EXE" create -y -p "$ENV_PREFIX" -c conda-forge --file "$HERE/../envs/sonic_deploy.conda.txt"
fi

fetch() {  # name -> verified archive path
  local archive url sha
  archive="$(pin "['$1']['archive']")"; url="$(pin "['$1']['url']")"; sha="$(pin "['$1']['sha256']")"
  if [ ! -f "$DOWNLOADS/$archive" ]; then
    curl -fL --retry 3 -o "$DOWNLOADS/$archive.part" "$url" >&2
    mv "$DOWNLOADS/$archive.part" "$DOWNLOADS/$archive"
  fi
  echo "$sha  $DOWNLOADS/$archive" | sha256sum -c - >&2
  echo "$DOWNLOADS/$archive"
}
TRT="$THIRD/TensorRT-$(pin "['tensorrt']['version']")"
ORT="$THIRD/onnxruntime-linux-x64-$(pin "['onnxruntime']['version']")"
BOOST="$THIRD/boost-headers-$(pin "['boost_headers']['version']")"
[ -d "$TRT/lib" ] || tar -xzf "$(fetch tensorrt)" -C "$THIRD"
[ -d "$ORT/lib" ] || tar -xzf "$(fetch onnxruntime)" -C "$THIRD"
if [ ! -d "$BOOST/include" ]; then  # .conda = zip holding pkg-*.tar.zst
  tmp="$(mktemp -d)"
  (cd "$tmp" && "$PYTHON" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$(fetch boost_headers)" \
    && mkdir -p "$BOOST" && tar --use-compress-program="$ENV_PREFIX/bin/zstd -d" -xf pkg-*.tar.zst -C "$BOOST")
  rm -rf "$tmp"
fi

export TensorRT_ROOT="$TRT" onnxruntime_ROOT="$ORT" CUDAToolkit_ROOT="$ENV_PREFIX" CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export HAS_ROS2=0
BUILD="${BUILD:-$PREFIX/build/gear_sonic_deploy}"

cmake -S "$DEPLOY" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_PREFIX_PATH="$ENV_PREFIX" \
  -DCMAKE_LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib;$ENV_PREFIX/lib" \
  -DCMAKE_INCLUDE_PATH="$ENV_PREFIX/targets/x86_64-linux/include;$ENV_PREFIX/include" \
  -DCMAKE_CXX_FLAGS="-isystem $BOOST/include" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$ENV_PREFIX/targets/x86_64-linux/lib -L$ENV_PREFIX/lib -Wl,-rpath-link,$ENV_PREFIX/lib"
cmake --build "$BUILD" -j"$JOBS" --target g1_deploy_onnx_ref
BIN="$DEPLOY/target/release/g1_deploy_onnx_ref"
test -x "$BIN"
echo
echo "export SONIC_DEPLOY_BIN=$BIN"
echo "export SONIC_DEPLOY_LIBRARY_PATH=$TRT/lib:$ORT/lib:$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib"
