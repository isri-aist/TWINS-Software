import gymnasium as gym
import numpy as np

import twin_arms_controller.envs  # noqa: F401


class OperationRealTwinArmsFollower:
    def __init__(self, **env_kwargs):
        self.env_kwargs = env_kwargs
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "twin_arms_controller/RealTwinArmsFollowerEnv-v0",
            **self.env_kwargs,
        )

    def reset(self):
        super().reset()
        self.motion_manager.reset()

    def record_data(self):
        super().record_data()

        env = self.env.unwrapped
        if not env.save_follower_debug_data:
            return

        command_joint_pos = np.concatenate(
            [
                self.motion_manager.get_command_data(key)
                for key in env.command_keys_for_step
            ]
        )
        command_joint_pos = env._clip_action(
            command_joint_pos + env.command_joint_pos_offset
        )
        raw_debug = env.get_motor_raw_debug_from_joint_pos(command_joint_pos)

        self.data_manager.append_single_data(
            env.COMMAND_DYNAMIXEL_RAW_GOAL, raw_debug["raw_goal"]
        )
        self.data_manager.append_single_data(
            env.COMMAND_DYNAMIXEL_RAW_GOAL_UNCLIPPED,
            raw_debug["raw_goal_unclipped"],
        )
        self.data_manager.append_single_data(
            env.MEASURED_DYNAMIXEL_RAW_POS,
            env.cached_measured_motor_raw_pos.copy(),
        )
        self.data_manager.append_single_data(
            env.DYNAMIXEL_POSITION_LIMIT_MIN,
            env.motor_position_limit_min.copy(),
        )
        self.data_manager.append_single_data(
            env.DYNAMIXEL_POSITION_LIMIT_MAX,
            env.motor_position_limit_max.copy(),
        )
        self.data_manager.append_single_data(
            env.MEASURED_DYNAMIXEL_PRESENT_CURRENT_RAW,
            env.cached_measured_motor_present_current.copy(),
        )
        self.data_manager.append_single_data(
            env.MEASURED_DYNAMIXEL_PRESENT_PWM_RAW,
            env.cached_measured_motor_present_pwm.copy(),
        )
        self.data_manager.append_single_data(
            env.DYNAMIXEL_CURRENT_LIMIT_RAW,
            env.motor_current_limit.copy(),
        )
        self.data_manager.append_single_data(
            env.DYNAMIXEL_PWM_LIMIT_RAW,
            env.motor_pwm_limit.copy(),
        )

    def run(self):
        try:
            super().run()
        finally:
            self.env.close()
