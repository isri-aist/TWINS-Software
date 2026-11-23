from robo_manip_baselines.bin.Rollout import RolloutMain

from twin_arms_controller.common import apply_twin_arms_data_patches


class RolloutMainTwinArmsController(RolloutMain):
    operation_parent_module_str = "twin_arms_controller.envs.operation"


def main():
    apply_twin_arms_data_patches()
    main = RolloutMainTwinArmsController()
    main.run()


if __name__ == "__main__":
    main()
