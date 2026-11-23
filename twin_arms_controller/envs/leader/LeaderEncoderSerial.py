import struct

import numpy as np
import serial


class LeaderEncoderSerial:
    SERIAL_BAUD_RATE = 38400
    NUMBER_OF_ENCODERS = 8
    LENGTH_FRAME_ENCODER = 20
    VALUE_START_OF_FRAME = 0xFF

    def __init__(self, device):
        self.device = device
        print(f"[{self.__class__.__name__}] Open encoder device: {device}")
        self.serial = serial.Serial(
            port=device,
            baudrate=self.SERIAL_BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            rtscts=True,
            timeout=None,
        )
        self.serial.reset_input_buffer()
        self.has_printed_read_wait = False

    def close(self):
        self.serial.close()

    @staticmethod
    def checksum(frame):
        value = 0
        for byte in frame[:-1]:
            value ^= byte
        return value

    def read_counts(self):
        if not self.has_printed_read_wait:
            print(
                f"[{self.__class__.__name__}] Wait for the first encoder frame: {self.device}"
            )
            self.has_printed_read_wait = True

        self.serial.reset_input_buffer()

        frame = bytearray()
        while True:
            while len(frame) != self.LENGTH_FRAME_ENCODER:
                chunk = self.serial.read(self.LENGTH_FRAME_ENCODER - len(frame))
                if not chunk:
                    raise RuntimeError(
                        f"[{self.__class__.__name__}] Failed to read from {self.device}"
                    )
                frame.extend(chunk)

            if frame[0] == self.VALUE_START_OF_FRAME and frame[-1] == self.checksum(
                frame
            ):
                _, *encoders = struct.unpack_from("<H8h", frame, 1)
                return np.array(encoders, dtype=np.float64)

            del frame[0]
