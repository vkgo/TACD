# TACD on a Unitree G1

Type a sentence, press Enter, and a Unitree G1 performs it. A resident server keeps the TACD model
([`weijinhuang/TACD-HY-Motion-Lite`](https://huggingface.co/weijinhuang/TACD-HY-Motion-Lite)) and a
UMR retargeting worker loaded. Each prompt is generated in 8 steps, retargeted to the G1 with UMR,
given a stand-in and a stand-out transition, and spliced into a continuous 50 Hz stream that NVIDIA's
SONIC whole-body controller tracks over ZMQ (Protocol v1). With nothing queued the stream is SONIC's
standing pose.

```
text ─▶ TACD (8 steps) ─▶ UMR ─▶ stand-in / stand-out ─▶ ZMQ stream ─▶ SONIC g1_deploy_onnx_ref ─▶ G1
```

The same clips can also be played in a single-process MuJoCo player or in IsaacLab.

## Install

Python environments (pinned lists in [`envs/`](envs)):

| env | python | used for |
|---|---|---|
| [`../requirements.txt`](../requirements.txt) | 3.12, torch 2.8.0 cu129 | server, player, SONIC launcher (the repository's main environment) |
| `envs/umr.txt` | 3.12, torch 2.4.1 cu121 | UMR worker (`UMR_PYTHON`) |
| `envs/sonic_sim.txt` | 3.10 | SONIC MuJoCo sim (`SONIC_SIM_PYTHON`) |
| `envs/sonic_deploy.conda.txt` | conda | building and running the SONIC deploy |

External code and data:

```bash
# SONIC: GR00T-WholeBodyControl 7f15131 + GEAR-SONIC sonic_v1_1 (versions and sha256 in sonic/pins.json)
bash sonic/install.sh ~/sonic            # add --isaaclab for the IsaacLab checkpoint
bash sonic/build_deploy.sh ~/sonic       # TensorRT 10.13.3.9, ONNX Runtime 1.16.3, g1_deploy_onnx_ref

# UMR at c56b630
git clone https://github.com/hanyang9/UMR ~/UMR && git -C ~/UMR checkout c56b6301ded02a187a30cc6aafa4f535735104d2

# HY-Motion 1.0 source (postprocessing and body model)
git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0 ~/HY-Motion-1.0
```

Put `SMPLX_NEUTRAL.npz` from [SMPL-X](https://smpl-x.is.tue.mpg.de) into `$UMR_ROOT/smpl/` (or set
`SMPLX_MODEL_DIR`). Then:

```bash
export PYTHONPATH=$PWD                      # this directory (humanoid/)
export SONIC_ROOT=~/sonic/GR00T-WholeBodyControl
export SONIC_SIM_PYTHON=<sonic_sim env>/bin/python
export SONIC_DEPLOY_LIBRARY_PATH=...        # printed by build_deploy.sh
export UMR_ROOT=~/UMR UMR_PYTHON=<umr env>/bin/python
export HY_MOTION_ROOT=~/HY-Motion-1.0
```

The first start downloads the TACD model and its text encoders from Hugging Face; `TACD_MODEL` points
to another copy. Caches (UMR correspondence, resolved robot XML) go to `~/.cache/tacd_humanoid`
(`TACD_HUMANOID_CACHE`). UMR's first start trains its correspondence for the shared body shape once.

## Start SONIC

Simulation (SONIC's MuJoCo sim and deploy on loopback; brings the robot up, checks that it stands, and
switches the deploy to ZMQ streaming as soon as the server's stream arrives):

```bash
python -m tacd_humanoid.stream.sonic_sim --run-dir runs/sonic --zmq-host localhost
```

Robot (from `$SONIC_ROOT/gear_sonic_deploy`, on the network interface connected to the robot):

```bash
LD_LIBRARY_PATH=$SONIC_DEPLOY_LIBRARY_PATH ./target/release/g1_deploy_onnx_ref <robot NIC> \
  policy/sonic_v1_1/model_decoder.onnx reference/example/ \
  --obs-config policy/sonic_v1_1/observation_config.yaml \
  --encoder-file policy/sonic_v1_1/model_encoder.onnx \
  --input-type zmq --zmq-host <server ip> --zmq-port 5556 --zmq-topic pose
```

Press `]` to start control; once the server is streaming, press `Enter` to switch to ZMQ streaming
mode (`ZMQ STREAMING MODE: ENABLED`), as in SONIC's ZMQ tutorial.

## Start the server

```bash
CUDA_VISIBLE_DEVICES=0 python -m tacd_humanoid.stream.server --bind tcp://0.0.0.0:5556 --out runs/session
```

It streams the standing pose right away, then loads TACD and the UMR worker.

## Type a prompt, press Enter

```
a person waves the right hand        ← generate (TACD ~0.6 s, UMR ~13 s on an RTX 3090)
                                     ← Enter: send (spliced into the stream 0.02 s later)
:d 6                                 ← duration of the next clip in seconds (default 4)
:s 3                                 ← seed of the next clip
:q                                   ← quit
```

Each clip is saved under `runs/session/NNN/` (`human.npz`, `reference_robot.npz`, `qpos50.npy`).
A clip sent while another plays starts when the robot is back in the standing pose.
`python -m tacd_humanoid.stream.evaluate_session --server-dir runs/session --sonic-dir runs/sonic`
scores a simulated session; `python -m tacd_humanoid.stream.sender --reference <clip>/reference_robot.npz --log sent.npz`
streams one saved clip.

## Single-process MuJoCo player

SONIC's encoder and decoder ONNX models and MuJoCo in one Python process, without the C++ deploy:

```bash
export ISAACLAB_PYTHON=<IsaacLab env>/bin/python   # runs SONIC's official CSV exporter
python -m tacd_humanoid.export --qpos runs/session/001/reference_robot.npz --out runs/player/001
python -m tacd_humanoid.mujoco_track --csv runs/player/001/csv/motion \
  --reference runs/session/001/reference_robot.npz --out runs/player/001/mujoco
```

Writes `trajectory.npz`, `mujoco_metrics.json` and a reference-vs-executed `video.mp4`.

## IsaacLab

Tracks the exported motion-lib clips with SONIC v1.1 through the official `gear_sonic/eval_agent_trl.py`
(Isaac Sim and IsaacLab installed per GR00T-WholeBodyControl; checkpoint from `sonic/install.sh --isaaclab`):

```bash
export SONIC_CHECKPOINT=~/sonic/sonic_v1_1/last.pt
python -m tacd_humanoid.sonic --motions runs/player/001/motionlib --out runs/isaac/001
```

## Licences

Apache License 2.0 ([`../LICENSE`](../LICENSE)); third-party files keep their licences, see
[THIRD_PARTY.md](THIRD_PARTY.md). Powered by Tencent HY.
