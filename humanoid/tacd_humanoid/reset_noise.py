"""Paired initial-joint perturbations at the official evaluation reset write."""
import numpy as np


def joint_noise(seed, amplitude, mujoco_to_isaac):
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("initial joint noise must be finite and nonnegative")
    return np.random.default_rng(seed).uniform(-amplitude, amplitude, 29)[mujoco_to_isaac]


def install_reset_noise(command, seed, amplitude, mujoco_to_isaac):
    """No hook or RNG draw for amplitude zero; scope the write hook to reset."""
    evidence = dict(seed=seed, amplitude_rad=amplitude, reset_calls=0, first_by_env={})
    if amplitude == 0:
        return evidence
    import torch
    noise = joint_noise(seed, amplitude, mujoco_to_isaac)
    evidence['requested_noise_isaaclab'] = noise.tolist()
    original_reset = command._resample_command
    robot = command.robot

    def reset(env_ids):
        original_write = robot.write_joint_state_to_sim
        def write(position, velocity, *args, **kwargs):
            perturb = torch.as_tensor(noise, dtype=position.dtype, device=position.device)
            actual = position + perturb
            ids = kwargs['env_ids']
            indices = ids.detach().cpu().tolist() if hasattr(ids, 'detach') else list(ids)
            nominal_cpu = position.detach().cpu().numpy()
            actual_cpu = actual.detach().cpu().numpy()
            for row, idx in enumerate(indices):
                if str(idx) not in evidence['first_by_env']:
                    evidence['first_by_env'][str(idx)] = dict(nominal=nominal_cpu[row].tolist(), actual=actual_cpu[row].tolist())
            evidence['reset_calls'] += 1
            return original_write(actual, velocity, *args, **kwargs)
        robot.write_joint_state_to_sim = write
        try:
            return original_reset(env_ids)
        finally:
            robot.write_joint_state_to_sim = original_write
    command._resample_command = reset
    return evidence
