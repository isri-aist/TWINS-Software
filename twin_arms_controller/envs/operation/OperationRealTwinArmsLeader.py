import gymnasium as gym

import twin_arms_controller.envs  # noqa: F401


class OperationRealTwinArmsLeader:
    def __init__(self, **env_kwargs):
        self.env_kwargs = env_kwargs
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "twin_arms_controller/RealTwinArmsLeaderEnv-v0",
            **self.env_kwargs,
        )

    def run(self):
        try:
            super().run()
        finally:
            self.env.close()
