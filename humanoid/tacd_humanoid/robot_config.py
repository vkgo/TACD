"""Read G1 armatures from SONIC's IsaacLab source without importing Isaac Sim."""
import ast
from pathlib import Path
import re

import numpy as np
from .paths import sonic_root


def g1_armatures(joint_names):
    source = sonic_root() / "gear_sonic/envs/manager_env/robots/g1.py"
    tree = ast.parse(source.read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("ARMATURE_"):
                constants[name] = ast.literal_eval(node.value)
            elif name == "G1_CYLINDER_MODEL_12_DEX_CFG":
                config = node.value
    actuators = next(k.value for k in config.keywords if k.arg == "actuators")
    values = {}
    for actuator in actuators.values:
        kwargs = {k.arg: k.value for k in actuator.keywords}
        patterns = ast.literal_eval(kwargs["joint_names_expr"])
        armature = eval(compile(ast.Expression(kwargs["armature"]), str(source), "eval"),
                        {"__builtins__": {}}, constants)
        for name in joint_names:
            if any(re.fullmatch(pattern, name) for pattern in patterns):
                values[name] = next(v for p, v in armature.items() if re.fullmatch(p, name)) if isinstance(armature, dict) else armature
    # IsaacLab stores actuator tensors in float32. Preserve those exact values
    # when assigning them to MuJoCo's float64 dynamics arrays.
    return np.asarray([values[name] for name in joint_names], dtype=np.float32)
