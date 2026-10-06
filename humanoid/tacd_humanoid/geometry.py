"""Lowest point of a collidable MuJoCo geometry."""
import numpy as np


def geom_floor(model, data, index):
    import mujoco
    kind = model.geom_type[index]
    size = model.geom_size[index]
    zrow = data.geom_xmat[index].reshape(3, 3)[2]
    center = data.geom_xpos[index, 2]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + abs(zrow[2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[0] * np.linalg.norm(zrow[:2]) + abs(zrow[2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(zrow) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.linalg.norm(zrow * size)
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[index]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return float(center + (model.mesh_vert[start:start + count] @ zrow).min())
    else:
        raise ValueError(f"Unsupported collidable geometry {kind}")
    return float(center - extent)
