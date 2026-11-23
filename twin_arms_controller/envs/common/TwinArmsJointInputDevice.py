from robo_manip_baselines.common import DataKey
from robo_manip_baselines.teleop import InputDeviceBase


class TwinArmsJointInputDevice(InputDeviceBase):
    def __init__(self, env, body_manager):
        super().__init__()

        self.env = env
        self.body_manager = body_manager

    def connect(self):
        self.connected = True

    def close(self):
        self.connected = False

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

    def is_ready(self):
        return True

    def set_command_data(self):
        if self.env.unwrapped.read_only:
            command_joint_pos = self.env.unwrapped.cached_joint_pos
        else:
            command_joint_pos = self.env.unwrapped.cached_command_joint_pos
        self.body_manager.set_command_data(DataKey.COMMAND_JOINT_POS, command_joint_pos)
