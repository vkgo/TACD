"""Instrument the official evaluator and optionally perturb initial reset joints."""
import json
import os
import time
from pathlib import Path

import numpy as np
from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback


class PolicyEvidenceCallback(ImEvalCallback):
    def _get_inference_policy(self, device=None):
        policy = super()._get_inference_policy(device)
        self.policy_calls = 0
        self.control_steps = 0
        self.physics_steps = 0
        self.action_abs_max = 0.0
        self.first_command_unix = None
        self.trajectory = []
        self.termination_causes = {}
        self.done_history = []
        self.motion_id_history = []
        self.pre_step_history = []
        self.control_step_indices = []
        self.terminal_states = {}
        from .reset_noise import install_reset_noise
        from .export import mappings
        amplitude = float(os.environ.get("TACD_INITIAL_JOINT_NOISE_RAD", "0"))
        self.reset_noise_evidence = install_reset_noise(self.env.motion_command,
            int(os.environ.get("TACD_PERTURB_SEED", "0")), amplitude,
            mappings()["G1_MUJOCO_TO_ISAACLAB_DOF"] if amplitude else None)
        self.first_policy_action = None
        if os.environ.get("TACD_RECORD_TRAJECTORY") == "1":
            manager = self.env.env.recorder_manager
            original_pre_reset = manager.record_pre_reset

            def capture_terminal(env_ids, *args, **kwargs):
                states = self._robot_states()
                for idx in env_ids.detach().cpu().tolist():
                    self.terminal_states[idx] = states[idx].copy()
                return original_pre_reset(env_ids, *args, **kwargs)
            manager.record_pre_reset = capture_terminal

        def measured(*args, **kwargs):
            action = policy(*args, **kwargs)
            if self.first_policy_action is None and amplitude:
                self.first_policy_action = action.detach().cpu().numpy().tolist()
            self.policy_calls += 1
            return action
        return measured

    def _robot_states(self):
        robot = self.env.env.scene["robot"].data
        pos = robot.root_pos_w - self.env.env.scene.env_origins
        return np.concatenate([pos.detach().cpu().numpy(), robot.root_quat_w.detach().cpu().numpy(),
                               robot.joint_pos.detach().cpu().numpy()], axis=-1)

    def env_step(self, actor_state):
        # Preserve motion identity before the simulator can reset a terminated env.
        ids = self.env._motion_lib.get_motion_ids_in_dataset(self.env.motion_ids).detach().cpu().tolist()
        recording = os.environ.get("TACD_RECORD_TRAJECTORY") == "1"
        if recording and not self.trajectory:
            self.trajectory.append(self._robot_states())
        if recording:
            self.pre_step_history.append(self._robot_states())
            self.control_step_indices.append(int(self.curr_steps))
        self.terminal_states = {}
        self.action_abs_max = max(self.action_abs_max, float(actor_state["actions"].abs().max().item()))
        if self.first_command_unix is None:
            self.first_command_unix = time.time()
        physics_before = self.env.env._sim_step_counter
        state = super().env_step(actor_state)
        self.control_steps += 1
        self.physics_steps += self.env.env._sim_step_counter - physics_before
        if recording:
            executed = self._robot_states()
            for idx, terminal in self.terminal_states.items():
                executed[idx] = terminal
            self.trajectory.append(executed)
            self.done_history.append(state["dones"].detach().cpu().numpy().copy())
            self.motion_id_history.append(ids)
        died = state["dones"].bool() & ~state["extras"]["time_outs"].bool()
        died &= self.curr_steps <= self.env._motion_lib.get_motion_num_steps(self.env.motion_ids) - 1
        manager = self.env.env.termination_manager
        for env_id in died.nonzero().flatten().cpu().tolist():
            key = str(self.env._motion_lib._motion_data_keys[ids[env_id]])
            if key not in self.termination_causes:
                terms = [name for name in manager.active_terms if bool(manager.get_term(name)[env_id])
                         and not manager.get_term_cfg(name).time_out]
                self.termination_causes[key] = dict(terms=terms, control_step=int(self.curr_steps),
                                                    seconds=float(self.curr_steps) / 50.)
        return state

    def save_metrics_eval(self, metrics_eval):
        super().save_metrics_eval(metrics_eval)
        out = Path(self.output_dir)
        evidence = dict(
            controller="SONIC v1.1", simulator="IsaacLab", policy_calls=self.policy_calls,
            control_steps=self.control_steps, num_envs=self.env.env.num_envs,
            control_decimation=self.env.env.cfg.decimation,
            physics_dt_seconds=self.env.env.cfg.sim.dt,
            physics_steps=self.physics_steps, action_abs_max=self.action_abs_max,
            first_command_unix=self.first_command_unix, kinematic_replay=False,
            checkpoint=os.environ["TACD_CHECKPOINT"],
            encoder=os.environ["TACD_ENCODER"],
            encoder_sample_weights=dict(self.env.motion_command.encoder_sample_probs_dict),
        )
        evidence["initial_joint_perturbation"] = self.reset_noise_evidence
        if self.first_policy_action is not None:
            evidence["first_policy_action"] = self.first_policy_action
        manager = self.env.env.termination_manager
        evidence["termination_terms"] = {
            name: dict(function=manager.get_term_cfg(name).func.__name__,
                       params=manager.get_term_cfg(name).params,
                       time_out=manager.get_term_cfg(name).time_out)
            for name in manager.active_terms
        }
        (out / "policy_evidence.json").write_text(json.dumps(evidence, indent=2, default=str))
        (out / "termination_causes.json").write_text(json.dumps(self.termination_causes, indent=2))
        if self.trajectory:
            np.savez_compressed(out / "executed_trajectory.npz", qpos=np.stack(self.trajectory),
                                dones=np.stack(self.done_history), dataset_motion_ids=np.asarray(self.motion_id_history),
                                dataset_motion_keys=np.asarray(self.env._motion_lib._motion_data_keys, dtype=str),
                                pre_step_qpos=np.stack(self.pre_step_history),
                                control_step_indices=np.asarray(self.control_step_indices),
                                fps=50.0, joint_order="isaaclab", quaternion_order="wxyz",
                                timing="Initial state then post-step states; terminal states captured before reset")
