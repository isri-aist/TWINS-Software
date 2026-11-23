from gymnasium.envs.registration import register

# Real
## LeaderArm
register(
    id="twin_arms_controller/RealTwinArmsLeaderEnv-v0",
    entry_point="twin_arms_controller.envs.leader:RealTwinArmsLeaderEnv",
)

## FollowerArm
register(
    id="twin_arms_controller/RealTwinArmsFollowerEnv-v0",
    entry_point="twin_arms_controller.envs.follower:RealTwinArmsFollowerEnv",
)
