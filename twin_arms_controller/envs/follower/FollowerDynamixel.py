import numpy as np
from dynamixel_sdk import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)


def u32_to_i32(value):
    return value - 0x100000000 if value & 0x80000000 else value


def u16_to_i16(value):
    return value - 0x10000 if value & 0x8000 else value


def i32_to_u32(value):
    return int(value) & 0xFFFFFFFF


def encode_u32_le(value):
    value = i32_to_u32(value)
    return [
        DXL_LOBYTE(DXL_LOWORD(value)),
        DXL_HIBYTE(DXL_LOWORD(value)),
        DXL_LOBYTE(DXL_HIWORD(value)),
        DXL_HIBYTE(DXL_HIWORD(value)),
    ]


class FollowerDynamixel:
    PROTOCOL_VERSION = 2.0
    LEN_4 = 4
    TORQUE_OFF = 0
    TORQUE_ON = 1
    POSITION_CONTROL_MODE = 3
    EXTENDED_POSITION_CONTROL_MODE = 4

    ADDR_BY_SERIES = {
        "x": {
            "torque_enable": 64,
            "goal_position": 116,
            "profile_acceleration": 108,
            "profile_velocity": 112,
            "present_position": 132,
            "present_pwm": 124,
            "present_current": 126,
            "position_p_gain": 84,
            "status_return_level": 68,
        },
        "p": {
            "torque_enable": 512,
            "goal_position": 564,
            "profile_acceleration": 556,
            "profile_velocity": 560,
            "present_position": 580,
            "present_pwm": 572,
            "present_current": 574,
            "status_return_level": 516,
        },
    }
    ADDR_OPERATING_MODE = 11
    ADDR_PWM_LIMIT = 36
    ADDR_CURRENT_LIMIT = 38
    ADDR_ACCELERATION_LIMIT = 40
    ADDR_VELOCITY_LIMIT = 44
    ADDR_MAX_POSITION_LIMIT = 48
    ADDR_MIN_POSITION_LIMIT = 52

    def __init__(
        self,
        device,
        motor_ids,
        series,
        baudrate=57600,
        enable_present_effort_read=False,
    ):
        if series not in self.ADDR_BY_SERIES:
            raise ValueError(f"Unsupported Dynamixel series: {series}")

        self.device = device
        self.motor_ids = list(motor_ids)
        self.series = series
        self.baudrate = baudrate
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)
        self.port_handler = PortHandler(device)
        self.addr = self.ADDR_BY_SERIES[series]
        self.enable_present_effort_read = enable_present_effort_read
        self.motor_limit_values = {
            motor_id: {"pwm_limit": np.nan, "current_limit": np.nan}
            for motor_id in self.motor_ids
        }

        print(
            f"[{self.__class__.__name__}] Opening {device} "
            f"for {series.upper()} series ids={self.motor_ids}"
        )
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open Dynamixel port: {device}")
        if not self.port_handler.setBaudRate(baudrate):
            self.port_handler.closePort()
            raise RuntimeError(f"Failed to set Dynamixel baudrate: {device}")

        self.group_sync_read = GroupSyncRead(
            self.port_handler,
            self.packet_handler,
            self.addr["present_position"],
            self.LEN_4,
        )
        for motor_id in self.motor_ids:
            if not self.group_sync_read.addParam(motor_id):
                raise RuntimeError(f"[ID:{motor_id:03d}] groupSyncRead addParam failed")
        if self.enable_present_effort_read:
            self.group_sync_read_effort = GroupSyncRead(
                self.port_handler,
                self.packet_handler,
                self.addr["present_pwm"],
                self.LEN_4,
            )
            for motor_id in self.motor_ids:
                if not self.group_sync_read_effort.addParam(motor_id):
                    raise RuntimeError(
                        f"[ID:{motor_id:03d}] groupSyncRead effort addParam failed"
                    )

    def close(self):
        self.port_handler.closePort()

    def setup_position_control(
        self,
        profile_velocity,
        profile_acceleration,
        pwm_limit=None,
        current_limit=None,
        position_limit_min=None,
        position_limit_max=None,
        position_p_gain=None,
        current_limit_motor_ids=None,
        motor_overrides=None,
    ):
        current_limit_motor_ids = (
            set(self.motor_ids)
            if current_limit_motor_ids is None
            else set(current_limit_motor_ids)
        )
        motor_overrides = {} if motor_overrides is None else motor_overrides
        operating_mode = (
            self.POSITION_CONTROL_MODE
            if self.series == "x"
            else self.EXTENDED_POSITION_CONTROL_MODE
        )
        self.disable_torque_all()
        for motor_id in self.motor_ids:
            motor_config = {
                "profile_velocity": profile_velocity,
                "profile_acceleration": profile_acceleration,
                "pwm_limit": pwm_limit,
                "current_limit": current_limit,
                "position_limit_min": position_limit_min,
                "position_limit_max": position_limit_max,
                "position_p_gain": position_p_gain,
            }
            motor_config.update(motor_overrides.get(motor_id, {}))
            self.write1(motor_id, self.addr["status_return_level"], 2)
            if (
                motor_config["position_limit_max"] is not None
                and motor_config["position_limit_min"] is not None
            ):
                self.write4(
                    motor_id,
                    self.ADDR_MAX_POSITION_LIMIT,
                    motor_config["position_limit_max"],
                )
                self.write4(
                    motor_id,
                    self.ADDR_MIN_POSITION_LIMIT,
                    motor_config["position_limit_min"],
                )
            self.write1(motor_id, self.ADDR_OPERATING_MODE, operating_mode)
            if motor_config["position_p_gain"] is not None:
                self.write2(
                    motor_id,
                    self.addr["position_p_gain"],
                    motor_config["position_p_gain"],
                )
            if motor_config["pwm_limit"] is not None:
                self.write2(motor_id, self.ADDR_PWM_LIMIT, motor_config["pwm_limit"])
            if (
                motor_config["current_limit"] is not None
                and motor_id in current_limit_motor_ids
            ):
                self.write2(
                    motor_id, self.ADDR_CURRENT_LIMIT, motor_config["current_limit"]
                )
            self.motor_limit_values[motor_id] = {
                "pwm_limit": float(motor_config["pwm_limit"])
                if motor_config["pwm_limit"] is not None
                else np.nan,
                "current_limit": float(motor_config["current_limit"])
                if (
                    motor_config["current_limit"] is not None
                    and motor_id in current_limit_motor_ids
                )
                else np.nan,
            }
            safe_profile_velocity = self._clip_profile_to_limit(
                motor_id,
                "velocity",
                motor_config["profile_velocity"],
                self.ADDR_VELOCITY_LIMIT,
            )
            safe_profile_acceleration = self._clip_profile_to_limit(
                motor_id,
                "acceleration",
                motor_config["profile_acceleration"],
                self.ADDR_ACCELERATION_LIMIT,
            )
            self.write4(motor_id, self.addr["profile_velocity"], safe_profile_velocity)
            self.write4(
                motor_id,
                self.addr["profile_acceleration"],
                safe_profile_acceleration,
            )

    def enable_torque_all(self):
        for motor_id in self.motor_ids:
            self.write1(motor_id, self.addr["torque_enable"], self.TORQUE_ON)

    def disable_torque_all(self):
        for motor_id in self.motor_ids:
            self.write1(motor_id, self.addr["torque_enable"], self.TORQUE_OFF)

    def sync_write_goal_positions(self, raw_goals):
        group_sync_write = GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            self.addr["goal_position"],
            self.LEN_4,
        )
        for motor_id, raw_goal in raw_goals.items():
            if not group_sync_write.addParam(motor_id, encode_u32_le(raw_goal)):
                raise RuntimeError(
                    f"{self.device} {self.series.upper()} [ID:{motor_id:03d}] "
                    "groupSyncWrite addParam failed"
                )

        result = group_sync_write.txPacket()
        group_sync_write.clearParam()
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"{self.device} {self.series.upper()} sync write goal positions "
                f"ids={sorted(raw_goals.keys())}: "
                f"{self.packet_handler.getTxRxResult(result)}"
            )

    def read_present_positions(self):
        result = None
        for _ in range(3):
            result = self.group_sync_read.txRxPacket()
            if result == COMM_SUCCESS:
                break
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"{self.device} {self.series.upper()} read present positions "
                f"ids={self.motor_ids}: {self.packet_handler.getTxRxResult(result)}"
            )

        positions = {}
        for motor_id in self.motor_ids:
            if not self.group_sync_read.isAvailable(
                motor_id, self.addr["present_position"], self.LEN_4
            ):
                raise RuntimeError(
                    f"{self.device} {self.series.upper()} [ID:{motor_id:03d}] "
                    "groupSyncRead getData failed"
                )
            raw = self.group_sync_read.getData(
                motor_id, self.addr["present_position"], self.LEN_4
            )
            positions[motor_id] = u32_to_i32(raw)
        return positions

    def read_present_efforts(self):
        if not self.enable_present_effort_read:
            raise RuntimeError(
                f"{self.device} {self.series.upper()} present effort read is disabled"
            )

        result = None
        for _ in range(3):
            result = self.group_sync_read_effort.txRxPacket()
            if result == COMM_SUCCESS:
                break
        if result != COMM_SUCCESS:
            raise RuntimeError(
                f"{self.device} {self.series.upper()} read present efforts "
                f"ids={self.motor_ids}: {self.packet_handler.getTxRxResult(result)}"
            )

        efforts = {}
        for motor_id in self.motor_ids:
            if not self.group_sync_read_effort.isAvailable(
                motor_id, self.addr["present_pwm"], self.LEN_4
            ):
                raise RuntimeError(
                    f"{self.device} {self.series.upper()} [ID:{motor_id:03d}] "
                    "groupSyncRead effort getData failed"
                )
            pwm = self.group_sync_read_effort.getData(
                motor_id, self.addr["present_pwm"], 2
            )
            current = self.group_sync_read_effort.getData(
                motor_id, self.addr["present_current"], 2
            )
            efforts[motor_id] = {
                "pwm": u16_to_i16(pwm),
                "current": u16_to_i16(current),
            }
        return efforts

    def read_position_limits(self):
        limits = {}
        for motor_id in self.motor_ids:
            limits[motor_id] = {
                "min": self.read4(motor_id, self.ADDR_MIN_POSITION_LIMIT),
                "max": self.read4(motor_id, self.ADDR_MAX_POSITION_LIMIT),
            }
        return limits

    def write1(self, motor_id, address, value):
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler,
            motor_id,
            address,
            int(value),
        )
        self._check_result(
            result,
            error,
            f"{self.device} {self.series.upper()} write1 ID:{motor_id} "
            f"addr:{address} value:{value}",
        )

    def write2(self, motor_id, address, value):
        result, error = self.packet_handler.write2ByteTxRx(
            self.port_handler,
            motor_id,
            address,
            int(value),
        )
        self._check_result(
            result,
            error,
            f"{self.device} {self.series.upper()} write2 ID:{motor_id} "
            f"addr:{address} value:{value}",
        )

    def write4(self, motor_id, address, value):
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler,
            motor_id,
            address,
            i32_to_u32(value),
        )
        self._check_result(
            result,
            error,
            f"{self.device} {self.series.upper()} write4 ID:{motor_id} "
            f"addr:{address} value:{value}",
        )

    def read4(self, motor_id, address):
        value, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id, address
        )
        self._check_result(
            result,
            error,
            f"{self.device} {self.series.upper()} read4 ID:{motor_id} "
            f"addr:{address}",
        )
        return u32_to_i32(value)

    def _clip_profile_to_limit(self, motor_id, profile_name, value, limit_address):
        if self.series != "p":
            return value

        limit = self.read4(motor_id, limit_address)
        if value <= limit:
            return value

        print(
            f"[{self.__class__.__name__}] "
            f"{self.device} P ID:{motor_id} profile {profile_name} "
            f"{value} exceeds limit {limit}; use {limit}."
        )
        return limit

    def _check_result(self, result, error, label):
        if result != COMM_SUCCESS:
            raise RuntimeError(f"{label}: {self.packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise RuntimeError(
                f"{label}: {self.packet_handler.getRxPacketError(error)}"
            )
