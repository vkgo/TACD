"""HY HumanMotion -> SMPL-X surface parameters for UMR (Z-up, meters, 30 fps).

HY local rotations use the SMPL-X body joint order and a T-pose rest frame, but the
HY wooden skeleton places joints differently from the SMPL-X regressor (several cm on
the torso and arms even after a shape fit). The bridge therefore:

1. uses one shared SMPL-X neutral shape (10 betas fitted once to HY motion joints and
   stored in assets/, because UMR caches one learned correspondence per template);
2. initializes every frame from HY's own local rotations (Y-up -> Z-up on the root);
3. refines pose/translation so the SMPL-X joints match HY joints_world, weighting limbs
   and end effectors over the internal spine/collar landmarks, with a pull toward the
   HY rotations to keep bone twist and temporal smoothness;
4. grounds the SMPL-X mesh once (global clip minimum at z=0), as HY does for its mesh.
"""
import functools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from .paths import ASSETS, hy_motion_root, smplx_model_dir

YUP_TO_ZUP = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
BETAS_FILE = ASSETS / "hy_smplx_betas.json"
JOINT_NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
               "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
               "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
# Internal torso landmarks sit at different depths in the two skeletons; they are not
# surface features and get a low weight so they do not drag limbs and end effectors.
LOW_WEIGHT = {"Spine1", "Spine2", "Spine3", "Neck", "L_Collar", "R_Collar"}
JOINT_WEIGHTS = np.array([0.1 if n in LOW_WEIGHT else 1.0 for n in JOINT_NAMES], dtype=np.float32)
FIT_CONFIG = dict(iters=200, lr=0.02, rot_weight=1e-2, temporal_weight=1.0, beta_weight=1e-4)


def hy_local_rotations(motion):
    import torch
    sys.path.insert(0, str(hy_motion_root()))
    from hymotion.utils.geometry import rot6d_to_rotation_matrix
    motion.validate()
    return rot6d_to_rotation_matrix(torch.from_numpy(motion.rot6d)).numpy()


def initial_rotations(motion):
    """HY local rotations as SMPL-X (T,22,3,3); the root is rotated from Y-up to Z-up."""
    local = hy_local_rotations(motion).astype(np.float32)
    local[:, 0] = YUP_TO_ZUP @ local[:, 0]
    return local


@functools.lru_cache(maxsize=4)
def smplx_model(device):
    """Loaded once per device; only its buffers are read (lbs), never modified."""
    import smplx
    return smplx.SMPLX(str(smplx_model_dir() / "SMPLX_NEUTRAL.npz"), use_pca=False, num_betas=10, flat_hand_mean=True).to(device)


def _forward(model, betas, rotations, transl):
    """SMPL-X LBS from (T,22,3,3) body rotations; face/hands at identity (flat hands)."""
    import torch
    from smplx.lbs import lbs
    t = len(rotations)
    eye = torch.eye(3, device=rotations.device).expand(t, 55 - 22, 3, 3)
    full = torch.cat([rotations, eye], 1)
    shape = torch.cat([betas[None].expand(t, -1), torch.zeros(t, model.num_expression_coeffs, device=rotations.device)], -1)
    shapedirs = torch.cat([model.shapedirs, model.expr_dirs], -1)
    vertices, joints = lbs(shape, full, model.v_template, shapedirs, model.posedirs, model.J_regressor,
                           model.parents, model.lbs_weights, pose2rot=False)
    return vertices + transl[:, None], joints[:, :22] + transl[:, None]


def _fit(model, target, rotations0, betas0, optimize_betas, iters, lr, rot_weight, temporal_weight, beta_weight):
    """Optimize small residual rotations R = R_hy @ exp(delta); no axis-angle wrap issues."""
    import torch
    from smplx.lbs import batch_rodrigues
    device = target.device
    weights = torch.tensor(JOINT_WEIGHTS, device=device)
    t = len(target)
    delta = torch.zeros(t, 22, 3, device=device, requires_grad=True)
    betas = betas0.clone().requires_grad_(optimize_betas)
    with torch.no_grad():
        pelvis = _forward(model, betas0, rotations0, torch.zeros_like(target[:, 0]))[1][:, 0]
    transl = (target[:, 0] - pelvis).clone().requires_grad_()
    rotations = lambda: rotations0 @ batch_rodrigues(delta.reshape(-1, 3)).reshape(t, 22, 3, 3)
    optimizer = torch.optim.Adam([delta, transl] + ([betas] if optimize_betas else []), lr=lr)
    for _ in range(iters):
        optimizer.zero_grad()
        joints = _forward(model, betas, rotations(), transl)[1]
        position = (((joints - target) ** 2).sum(-1) * weights).sum(-1).mean() / weights.sum()
        prior = (delta ** 2).mean()
        smooth = ((delta[1:] - delta[:-1]) ** 2).mean() if t > 1 else 0.
        loss = position + rot_weight * prior + temporal_weight * smooth + beta_weight * (betas ** 2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return rotations(), transl.detach(), betas.detach(), delta.detach().norm(dim=-1)


def fit_shared_betas(motions, device="cpu", iters=300):
    """Fit one SMPL-X shape to the fixed HY skeleton across several motions."""
    import torch
    model = smplx_model(device)
    rotations = np.concatenate([initial_rotations(m) for m in motions])
    targets = np.concatenate([m.joints_world @ YUP_TO_ZUP.T for m in motions])
    tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    # Concatenated clips: disable temporal smoothness so clip boundaries do not interact.
    return _fit(model, tensor(targets), tensor(rotations), torch.zeros(10, device=device), True,
                iters, 0.02, 1e-2, 0.0, 1e-4)[2].cpu().numpy().astype(np.float32)


def load_betas():
    return np.asarray(json.loads(BETAS_FILE.read_text())["betas"], dtype=np.float32)


def human_to_smplx(motion, device="cpu", betas=None, fit=True):
    """Return SMPL-X parameters, grounded vertices and FK validation against HY joints.

    fit=True: per-clip residual pose fit to the HY joints (FIT_CONFIG). fit=False ("direct"):
    the HY local rotations drive SMPL-X unchanged; only the translation puts the SMPL-X pelvis
    on the HY pelvis in every frame."""
    import torch
    model = smplx_model(device)
    betas = load_betas() if betas is None else np.asarray(betas, dtype=np.float32)
    tensor = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
    rotations0 = tensor(initial_rotations(motion))
    target = tensor(motion.joints_world @ YUP_TO_ZUP.T)
    cfg = FIT_CONFIG if fit else dict(iters=0)
    if fit:
        rotations, transl, _, change = _fit(model, target, rotations0, tensor(betas), False, cfg["iters"], cfg["lr"],
                                            cfg["rot_weight"], cfg["temporal_weight"], cfg["beta_weight"])
    else:
        with torch.no_grad():
            pelvis = _forward(model, tensor(betas), rotations0, torch.zeros_like(target[:, 0]))[1][:, 0]
        rotations, transl, change = rotations0, target[:, 0] - pelvis, torch.zeros(target.shape[:2], device=device)
    with torch.no_grad():
        vertices, joints = _forward(model, tensor(betas), rotations, transl)
        _, direct = _forward(model, tensor(betas), rotations0, torch.zeros_like(transl))
    vertices, joints, target = vertices.cpu().numpy(), joints.cpu().numpy(), target.cpu().numpy()
    direct = direct.cpu().numpy()
    direct += (target[:, 0] - direct[:, 0])[:, None]
    error = np.linalg.norm(joints - target, axis=-1)
    direct_error = np.linalg.norm(direct - target, axis=-1)
    ground = float(vertices[..., 2].min())
    shift = np.array([0., 0., ground], dtype=np.float32)
    poses = np.zeros((len(joints), 55, 3), dtype=np.float32)
    poses[:, :22] = Rotation.from_matrix(rotations.cpu().numpy().reshape(-1, 3, 3).astype(np.float64)).as_rotvec().reshape(-1, 22, 3)
    limbs = JOINT_WEIGHTS == 1
    validation = dict(
        median_m=float(np.median(error)), max_m=float(error.max()),
        limb_median_m=float(np.median(error[:, limbs])), limb_max_m=float(error[:, limbs].max()),
        direct_median_m=float(np.median(direct_error)), direct_max_m=float(direct_error.max()),
        ground_shift_m=ground, rotation_change_max_rad=float(change.max()),
        per_joint={name: dict(median_m=float(np.median(error[:, i])), max_m=float(error[:, i].max()))
                   for i, name in enumerate(JOINT_NAMES)},
        fit=cfg, betas=betas.tolist())
    return dict(poses=poses, trans=(transl.cpu().numpy() - shift).astype(np.float32), betas=betas,
                validation=validation, vertices=vertices - shift, joints=joints - shift)


def write_umr_npz(params, path):
    """UMR SMPL-X npz contract: poses (T,55,3), trans, betas, gender, fps, output_up=z."""
    np.savez(path, poses=params["poses"], trans=params["trans"], betas=params["betas"],
             gender=np.asarray("neutral"), mocap_frame_rate=np.asarray(30), output_up=np.asarray("z"))
