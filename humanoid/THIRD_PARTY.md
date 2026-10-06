# Third-party components

## NVIDIA GR00T-WholeBodyControl / GEAR-SONIC

- Code: https://github.com/NVlabs/GR00T-WholeBodyControl at `7f151314d4d1606544bf249d2a7a1cb754c64582`,
  Apache License 2.0, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Installed by
  `sonic/install.sh` as an external checkout; not redistributed here.
- Weights: `nvidia/GEAR-SONIC` at revision `6733128a3d8a523b1418b06bca3cdf61c8b0987f` (`sonic_v1_1`,
  `planner_sonic.onnx`), downloaded by `sonic/install.sh` and checked against the sha256 values in
  `sonic/pins.json`; not redistributed here.
  Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
- `tacd_humanoid/vendor/sonic_singleprocess.py`: written for this module; its policy constants, joint
  orderings and observation layout are ported from the GR00T-WholeBodyControl C++ deploy
  (`gear_sonic_deploy`, `policy_parameters.hpp`). Apache License 2.0; the file carries the NVIDIA
  copyright notice and a statement of the changes. Full text: `licenses/Apache-2.0.txt`.
- `tacd_humanoid/stream/sonic_sim_headless.py` drives the upstream simulator
  (`gear_sonic/scripts/run_sim_loop.py` classes) imported from the checkout.
- `tacd_humanoid/assets/g1_boxfeet.xml`: derived from the public
  `decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.xml` (Unitree G1 model, BSD-3-Clause,
  distributed in GR00T-WholeBodyControl under Apache-2.0). The four collision spheres under each foot
  are replaced by one box (`size 0.085 0.03 0.005`, `pos 0.035 0 -0.03`); the meshes are read from the
  checkout at run time.
- The deploy build uses TensorRT 10.13.3.9 (NVIDIA), ONNX Runtime 1.16.3 (MIT) and conda-forge
  `libboost-headers` 1.90.0 (Boost Software License), downloaded by `sonic/build_deploy.sh`.

## UMR

- https://github.com/hanyang9/UMR at `c56b6301ded02a187a30cc6aafa4f535735104d2`, MIT License,
  Copyright (c) 2026 Hanyang Cao and UMR contributors. External dependency (`UMR_ROOT`).
- `tacd_humanoid/umr_worker.py` contains a modified copy of `compute_robot_self_penetration_rows()`
  from `scripts/smpl_surface_retarget_common.py`; the MIT notice is reproduced in that file and in
  `licenses/UMR-MIT.txt`.
- `tacd_humanoid/assets/umr_tuned/g1_29dof_collisionproxy30.xml` is UMR's G1 model (Unitree
  BSD-3-Clause, distributed in UMR under MIT) with 3 cm sphere collision proxies on the hands and
  8 explicit `<exclude>` pairs. `umr_tuned.json` holds the retargeting settings.

## SMPL-X

- The SMPL-X neutral body model is not redistributed; download it from https://smpl-x.is.tue.mpg.de
  under its own licence. `tacd_humanoid/assets/hy_smplx_betas.json` holds ten SMPL-X shape
  coefficients fitted to the HY-Motion body.

## HY-Motion / TACD

- The TACD model (`weijinhuang/TACD-HY-Motion-Lite`) and the HY-Motion 1.0 code used for
  postprocessing (`HY_MOTION_ROOT`) are released under the Tencent HY-MOTION 1.0 Community License
  Agreement (`../hf_repo/LICENSE`).
  Tencent HY-MOTION 1.0 is licensed under the Tencent HY-MOTION 1.0 Community License Agreement,
  Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of “Tencent HY” are owned by
  Tencent or its affiliate. Powered by Tencent HY.

## IsaacLab

- The IsaacLab path runs the official `gear_sonic/eval_agent_trl.py` in an Isaac Sim / IsaacLab
  environment installed by the user (IsaacLab: BSD-3-Clause; Isaac Sim: NVIDIA licence).
