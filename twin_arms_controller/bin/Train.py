import importlib

from robo_manip_baselines.bin.Train import TrainMain

from twin_arms_controller.common import TwinArmsTrainMixin, apply_twin_arms_data_patches


class TrainMainTwinArmsController(TrainMain):
    def run(self):
        from robo_manip_baselines.common import camel_to_snake, remove_prefix

        policy_module = importlib.import_module(
            f"{self.policy_parent_module_str}.{camel_to_snake(self.args.policy)}"
        )
        TrainPolicyClass = getattr(policy_module, f"Train{self.args.policy}")

        class Train(TwinArmsTrainMixin, TrainPolicyClass):
            @property
            def policy_name(self):
                return remove_prefix(TrainPolicyClass.__name__, "Train")

        train = Train()
        train.run()
        train.close()


def main():
    apply_twin_arms_data_patches()
    main = TrainMainTwinArmsController()
    main.run()


if __name__ == "__main__":
    main()
