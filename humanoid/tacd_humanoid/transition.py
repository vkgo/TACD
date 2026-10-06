"""Stand-in (v2g) and stand-out (s2g) transitions between SONIC's standing pose and a clip."""
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .motion import RobotMotion

DEFAULT_STAND_VARIANT = "v2g"
DEFAULT_STAND_SUFFIX = "s2g"


@lru_cache(maxsize=1)
def box_feet_model():
    """Read-only FK model; each caller owns its separate MjData."""
    import mujoco
    from .vendor.sonic_singleprocess import DEFAULT_XML
    return mujoco.MjModel.from_xml_path(str(DEFAULT_XML))


def _foot_box_ids(model):
    import mujoco
    ids=[]
    for side in ['left','right']:
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
        ids.append(next(g for g in range(model.ngeom) if model.geom_bodyid[g]==body
            and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX
            and (model.geom_contype[g] or model.geom_conaffinity[g])))
    return ids


def foot_soles(qpos):
    """Lowest point across both official foot boxes, one value per pose."""
    import mujoco
    from .geometry import geom_floor
    model=box_feet_model();data=mujoco.MjData(model);ids=_foot_box_ids(model)
    result=[]
    for q in np.atleast_2d(qpos):
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        result.append(min(geom_floor(model,data,g) for g in ids))
    return np.asarray(result)


def ground_prefix(reference):
    """Ground only a v2-family prefix, with the untouched raw frame zero as endpoint."""
    n=int(reference.metadata['stand_prefix_frames'])
    settings=reference.metadata['stand_transition']
    hold=settings['hold_frames'];count=settings['transition_frames'];fps=reference.fps
    q=reference.qpos[:n+1].copy()
    soles=foot_soles(q)
    alpha=min_jerk(np.clip((np.arange(n+1)-hold)/count,0.,1.))
    target=alpha*soles[-1]
    exact=target-soles
    # Keep the standing hold and the untouched source endpoint exact. Their
    # computed correction differs from zero only by FK floating-point noise.
    exact[:hold+1]=0.;exact[-1]=0.
    # A supported one-interval transition has no second difference to estimate.
    peak=lambda z: float(np.max(np.abs(np.diff(z,n=2)))*fps**2) if len(z)>=3 else None
    original_peak=peak(q[:,2]);correction_peak=peak(exact)
    unsmoothed_peak=peak(q[:,2]+exact)
    # "Much larger" is explicit: >3x baseline, with a 1 m/s^2 floor for
    # near-constant original z. Five frames span four 30 Hz intervals.
    threshold=max(3.*original_peak,1.) if original_peak is not None else None
    correction=exact.copy();window=0
    if threshold is not None and unsmoothed_peak>threshold:
        kernel=np.array([1.,4.,6.,4.,1.])/16
        smoothed=np.convolve(np.pad(exact,(2,2),mode='edge'),kernel,mode='valid')
        # Projection can only raise the sole; endpoint/hold remain fixed.
        smoothed=np.maximum(smoothed,exact)
        smoothed[:hold+1]=0.;smoothed[-1]=0.
        correction=smoothed;window=5
    final=q[:,2]+correction
    result=RobotMotion(reference.qpos.copy(),fps,dict(reference.metadata),
                       None if reference.source_qpos is None else reference.source_qpos.copy())
    result.qpos[:n,2]=final[:n]
    grounded_soles=foot_soles(result.qpos[:n+1])
    residual=grounded_soles-target
    if np.min(residual)<-1e-6:
        raise ValueError('Grounded prefix falls below its prescribed sole target')
    result.metadata['grounding']=dict(method='Per-frame pelvis-z correction to min-jerk raw-frame-zero sole target',
        raw_frame0_sole_m=float(soles[-1]),max_abs_correction_m=float(np.max(np.abs(correction[:n]))),
        mean_abs_correction_m=float(np.mean(np.abs(correction[:n]))),
        mean_signed_correction_m=float(np.mean(correction[:n])),smoothing_window_frames=window,
        smoothing_kernel=[1/16,4/16,6/16,4/16,1/16] if window else None,
        acceleration_ratio=3.,acceleration_floor_m_s2=1.,acceleration_threshold_m_s2=threshold,
        original_peak_z_acceleration_m_s2=original_peak,
        correction_peak_z_acceleration_m_s2=correction_peak,
        unsmoothed_peak_z_acceleration_m_s2=unsmoothed_peak,
        final_peak_z_acceleration_m_s2=peak(final),
        final_correction_peak_z_acceleration_m_s2=peak(correction),
        minimum_sole_minus_target_m=float(np.min(residual)),
        maximum_sole_above_target_m=float(np.max(residual)),
        endpoint_correction_m=float(correction[-1]),
        acceleration_window='Prefix plus untouched raw frame zero; excludes generated motion velocity',
        boundary='Hold and source endpoint pinned to zero correction; post-smoothing target projection only raises z')
    result.validate()
    return result


@lru_cache(maxsize=1)
def standing_pose():
    """Default joints and upright root with the official box feet on z=0."""
    import mujoco
    from .vendor.sonic_singleprocess import DEFAULT_ANGLES_MUJOCO, DEFAULT_XML
    from .geometry import geom_floor
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)
    q = np.zeros(36); q[3] = 1.; q[7:] = DEFAULT_ANGLES_MUJOCO
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side + "_ankle_roll_link")
            for side in ["left", "right"]}
    floors = [geom_floor(model, data, g) for g in range(model.ngeom)
              if model.geom_bodyid[g] in feet and (model.geom_contype[g] or model.geom_conaffinity[g])]
    q[2] = -min(floors)
    return q


def min_jerk(u):
    u = np.asarray(u)
    return 10*u**3 - 15*u**4 + 6*u**5


def generated_time(playback_seconds, mapping=None):
    """Map time since prefix end to the generated motion clock (identity for v2g/s2g)."""
    if mapping and mapping.get("kind", "identity") != "identity":
        raise ValueError(f"Unsupported generated-time mapping {mapping['kind']!r}")
    return np.maximum(np.asarray(playback_seconds, dtype=float), 0.)


def generated_source_qpos(reference):
    if reference.source_qpos is not None:
        return reference.source_qpos
    return reference.qpos[int(reference.metadata.get("stand_prefix_frames", 0)):]


def foot_box_centers(qpos):
    """World-space centers of the official left/right collision foot boxes."""
    import mujoco
    model = box_feet_model(); data = mujoco.MjData(model)
    ids = _foot_box_ids(model)
    values = np.asarray(qpos); scalar = values.ndim == 1
    result = []
    for q in np.atleast_2d(values):
        data.qpos[:] = q; mujoco.mj_kinematics(model, data)
        result.append(data.geom_xpos[ids].copy())
    return np.asarray(result)[0] if scalar else np.asarray(result)


def fitted_standing_pose(first):
    """Two-point 2D Procrustes without scaling or reflection, preserving foot labels."""
    stand = standing_pose().copy()
    source = foot_box_centers(stand)[:, :2]; target = foot_box_centers(first)[:, :2]
    a = source-source.mean(0); b = target-target.mean(0)
    yaw = np.arctan2(np.sum(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]), np.sum(a*b))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translation = target.mean(0)-source.mean(0)@rotation.T
    stand[:2] = translation
    stand[3:7] = Rotation.from_rotvec([0., 0., yaw]).as_quat()[[3, 0, 1, 2]]
    residual = np.linalg.norm(source@rotation.T+translation-target, axis=1)
    return stand, dict(method="2D foot-box-center Procrustes; no scale/reflection",
        yaw_rad=float(yaw), root_xy_m=translation.tolist(), foot_residual_m=residual.tolist(),
        foot_residual_max_m=float(residual.max()), source_centers_xy=source.tolist(),
        target_centers_xy=target.tolist(), fitted_centers_xy=(source@rotation.T+translation).tolist())


def prepend_stand_transition(robot: RobotMotion, hold_s=.5, v_peak=1.5,
                             T_min=1., T_max=3., k=1.875, variant=DEFAULT_STAND_VARIANT,
                             foot_speed=.15) -> RobotMotion:
    """v2: hold the standing pose fitted under the first frame's feet, then a min-jerk transition
    into frame zero; v2g additionally grounds the prefix feet."""
    robot.validate()
    if variant not in ["none", "v2", "v2g"]:
        raise ValueError("Unknown standing-prefix variant")
    if hold_s < 0 or v_peak <= 0 or not 0 < T_min <= T_max or k <= 0 or foot_speed <= 0:
        raise ValueError("Invalid standing-prefix timing settings")
    if variant == "none":
        return RobotMotion(robot.qpos.copy(), robot.fps, {**robot.metadata,
            "stand_variant": "none", "stand_prefix_frames": 0, "stand_prefix_seconds": 0.})
    if variant=="v2g":
        reference=prepend_stand_transition(robot,hold_s,v_peak,T_min,T_max,k,"v2",foot_speed)
        result=ground_prefix(reference)
        result.metadata['stand_variant']=variant
        result.metadata['stand_transition']={**reference.metadata['stand_transition'],'variant':variant}
        return result
    fps = robot.fps; first = robot.qpos[0]
    n_hold = int(np.ceil(hold_s*fps-1e-10))
    stand, placement = fitted_standing_pose(first)
    delta = float(np.max(np.abs(first[7:]-stand[7:])))
    joint_T = float(np.clip(k*delta/v_peak, T_min, T_max))
    foot_T = placement["foot_residual_max_m"]/foot_speed
    # The explicitly required foot-duration floor may exceed the joint-duration cap.
    n_transition = int(np.ceil(max(joint_T, foot_T)*fps-1e-10)); T = n_transition/fps
    alpha = min_jerk(np.arange(n_transition)/n_transition)
    transition = stand+(first-stand)*alpha[:, None]
    rotations = Rotation.from_quat(np.stack([stand[3:7], first[3:7]])[:, [1, 2, 3, 0]])
    transition[:, 3:7] = Slerp([0, 1], rotations)(alpha).as_quat()[:, [3, 0, 1, 2]]
    prefix = np.concatenate([np.tile(stand, (n_hold, 1)), transition])
    mapping = dict(kind="identity")
    tail = robot.qpos; source = None
    metadata = {**robot.metadata, "stand_variant": variant, "stand_prefix_frames": len(prefix),
        "stand_prefix_seconds": len(prefix)/fps, "hold_s": n_hold/fps, "T": T,
        "generated_time_mapping": mapping, "original_frames": len(robot.qpos),
        "stand_transition": dict(enabled=True, variant=variant, hold_s_requested=hold_s,
            hold_frames=n_hold, transition_frames=n_transition, v_peak=v_peak, T_min=T_min,
            T_max=T_max, k=k, foot_speed_m_s=foot_speed, joint_duration_s=joint_T,
            foot_duration_floor_s=foot_T, duration_exceeds_joint_cap=T>T_max+1e-9,
            endpoint_velocity="zero",
            placement=placement, stand_root_height_m=float(stand[2]))}
    result = RobotMotion(np.concatenate([prefix.astype(robot.qpos.dtype), tail]), fps, metadata, source)
    result.validate()
    return result


def append_stand_suffix(reference: RobotMotion, *, variant=DEFAULT_STAND_SUFFIX, suffix_hold_s=1.,
                        v_peak=1.5, T_min=1., T_max=3., foot_speed=.15, k=1.875):
    """s2g: the time-reversed grounded v2g transition from the last frame back to standing."""
    reference.validate()
    if variant not in ["none", "s2g"]:
        raise ValueError("Unknown standing-suffix variant")
    if reference.metadata.get("stand_suffix_frames",0):
        raise ValueError("Reference already has a standing suffix")
    source=generated_source_qpos(reference).copy()
    metadata={**reference.metadata,"stand_suffix_variant":variant,
              "generated_end_seconds":(len(reference.qpos)-1)/reference.fps}
    if variant=="none":
        metadata.update(stand_suffix_frames=0,stand_suffix_seconds=0.)
        return RobotMotion(reference.qpos.copy(),reference.fps,metadata,source)
    # Reversal mirrors both Procrustes placement and the exact grounding target:
    # s(1-u)=1-s(u). The standing hold becomes the final hold.
    endpoint=RobotMotion(np.tile(reference.qpos[-1],(2,1)),reference.fps)
    mirror=prepend_stand_transition(endpoint,variant="v2g",hold_s=suffix_hold_s,
        v_peak=v_peak,T_min=T_min,T_max=T_max,foot_speed=foot_speed,k=k)
    n=mirror.metadata["stand_prefix_frames"]
    suffix=mirror.qpos[:n+1][::-1][1:].copy()
    np.testing.assert_array_equal(mirror.qpos[n],reference.qpos[-1])
    settings=mirror.metadata["stand_transition"]
    metadata.update(stand_suffix_frames=n,stand_suffix_seconds=n/reference.fps,
        stand_suffix_start_seconds=metadata["generated_end_seconds"],
        stand_suffix=dict(**{key:value for key,value in settings.items() if key!="variant"},
            variant=variant,hold_s=settings["hold_frames"]/reference.fps,
            hold_start_seconds=metadata["generated_end_seconds"]+settings["transition_frames"]/reference.fps,
            final_standing_pose=mirror.qpos[0].tolist(),
            grounding={**mirror.metadata["grounding"],
                "method":"Time-reversed v2g pelvis-z correction; target=(1-min_jerk(u))*raw_last_sole",
                "raw_last_frame_sole_m":mirror.metadata["grounding"]["raw_frame0_sole_m"],
                "acceleration_window":"Raw last frame plus suffix; excludes generated motion velocity",
                "boundary":"Raw last frame and final standing hold pinned to zero correction"}))
    result=RobotMotion(np.concatenate([reference.qpos,suffix]),reference.fps,metadata,source)
    result.validate()
    return result
