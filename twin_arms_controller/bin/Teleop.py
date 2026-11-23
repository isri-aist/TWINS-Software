from robo_manip_baselines.bin.Teleop import TeleopMain

from twin_arms_controller.common import apply_twin_arms_data_patches


class TeleopMainTwinArmsController(TeleopMain):
    operation_parent_module_str = "twin_arms_controller.envs.operation"


def main():
    apply_twin_arms_data_patches()
    main = TeleopMainTwinArmsController()
    main.run()


if __name__ == "__main__":
    main()
