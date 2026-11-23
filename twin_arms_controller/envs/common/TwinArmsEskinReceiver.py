import json
import struct
import time

import numpy as np
import zmq

MEASURED_ESKIN_KEY = "measured_eskin"


class TwinArmsEskinReceiver:
    ELEMENTS_PER_CELL = 17
    CELL_KEYS = (
        "cell_id",
        "patch_id",
        "segment_id",
        "position_x",
        "position_y",
        "position_z",
        "orientation_x",
        "orientation_y",
        "orientation_z",
        "orientation_w",
        "force1",
        "force2",
        "force3",
        "prox",
        "acc_x",
        "acc_y",
        "acc_z",
    )

    def __init__(self, endpoint, patch_names, timeout=1.0):
        if not patch_names:
            raise ValueError("eskin_patch_names must be non-empty.")

        self.endpoint = endpoint
        self.patch_names = list(patch_names)
        self.timeout = float(timeout)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.connect(endpoint)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.cell_counts = None
        self.vector_dim = None
        self.latest_vector = None
        self.latest_result = None
        self.last_receive_time = None

        self.latest_vector = self._receive_initial()

    def close(self):
        self.poller.unregister(self.socket)
        self.socket.close(linger=0)
        self.context.term()

    def read(self):
        vector = self._receive_latest(timeout=0.0)
        if vector is not None:
            return vector.copy()

        if time.time() - self.last_receive_time > self.timeout:
            raise RuntimeError(
                f"[{self.__class__.__name__}] No eSkin data received for "
                f"{self.timeout:.3f} s from {self.endpoint}."
            )
        return self.latest_vector.copy()

    def _receive_initial(self):
        start_time = time.time()
        while True:
            remaining_time = self.timeout - (time.time() - start_time)
            if remaining_time <= 0.0:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Failed to receive initial eSkin "
                    f"data within {self.timeout:.3f} s from {self.endpoint}."
                )

            vector = self._receive_latest(timeout=remaining_time)
            if vector is not None:
                return vector.copy()

    def _receive_latest(self, timeout):
        latest_result = None
        timeout_ms = int(1000.0 * timeout)
        events = dict(self.poller.poll(timeout_ms))
        if self.socket not in events:
            return None

        while self.socket in events:
            result = self._parse_zmq_message(self.socket.recv_multipart())
            if result is not None:
                latest_result = result
            events = dict(self.poller.poll(0))

        if latest_result is None:
            return None

        vector = self._make_vector(latest_result)
        self.latest_vector = vector.copy()
        self.latest_result = latest_result
        self.last_receive_time = time.time()
        return vector

    def _make_vector(self, result):
        missing_patch_names = [
            patch_name for patch_name in self.patch_names if patch_name not in result
        ]
        if missing_patch_names:
            raise RuntimeError(
                f"[{self.__class__.__name__}] eSkin patches not found: "
                + ", ".join(missing_patch_names)
            )

        prox_values = []
        force_values = []
        cell_counts = []
        for patch_name in self.patch_names:
            cells = result[patch_name]["cells"]
            cell_counts.append(len(cells))
            prox_values.extend(cell["prox"] for cell in cells)
            force_values.extend(
                cell["force1"] + cell["force2"] + cell["force3"] for cell in cells
            )

        if self.cell_counts is None:
            self.cell_counts = tuple(cell_counts)
            self.vector_dim = 2 * sum(self.cell_counts)
        elif tuple(cell_counts) != self.cell_counts:
            raise RuntimeError(
                f"[{self.__class__.__name__}] eSkin cell counts changed: "
                f"{tuple(cell_counts)} != {self.cell_counts}."
            )

        vector = np.asarray(prox_values + force_values, dtype=np.float64)
        if vector.shape[0] != self.vector_dim:
            raise RuntimeError(
                f"[{self.__class__.__name__}] eSkin vector dim changed: "
                f"{vector.shape[0]} != {self.vector_dim}."
            )
        return vector

    @classmethod
    def _parse_zmq_message(cls, msg_parts):
        if len(msg_parts) == 1 and len(msg_parts[0]) == 0:
            return None

        if len(msg_parts) == 2:
            return cls._parse_message(msg_parts[0], msg_parts[1])

        if len(msg_parts) != 1:
            raise ValueError(f"expected 1-part or 2-part message, got {len(msg_parts)}")

        msg = msg_parts[0]
        if len(msg) < 4:
            raise ValueError(f"message is too short: {len(msg)} bytes")

        metadata_size = struct.unpack("<I", msg[:4])[0]
        metadata_start = 4
        metadata_end = metadata_start + metadata_size
        if metadata_end > len(msg):
            raise ValueError("metadata is shorter than encoded size")

        return cls._parse_message(msg[metadata_start:metadata_end], msg[metadata_end:])

    @classmethod
    def _parse_message(cls, metadata_bytes, data_bytes):
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        result = {}
        offset_cells = 0
        for patch in metadata["patches"]:
            config_name = patch["config"]
            cell_count = int(patch["cell_count"])
            cells = cls._parse_cells(data_bytes, offset_cells, cell_count)
            offset_cells += cell_count
            result[config_name] = {
                "frame_id": patch.get("frame_id", ""),
                "cell_count": cell_count,
                "cells": cells,
            }

        expected_bytes = offset_cells * cls.ELEMENTS_PER_CELL * 4
        if expected_bytes != len(data_bytes):
            raise ValueError(
                f"unexpected data size: expected {expected_bytes} bytes, "
                f"got {len(data_bytes)} bytes"
            )
        return result

    @classmethod
    def _parse_cells(cls, raw_bytes, offset_cells, cell_count):
        start = offset_cells * cls.ELEMENTS_PER_CELL * 4
        end = start + cell_count * cls.ELEMENTS_PER_CELL * 4
        if end > len(raw_bytes):
            raise ValueError(
                f"message is shorter than metadata: need {end} bytes, "
                f"got {len(raw_bytes)} bytes"
            )

        values = struct.unpack(
            f"<{cell_count * cls.ELEMENTS_PER_CELL}f", raw_bytes[start:end]
        )
        cells = []
        for cell_idx in range(cell_count):
            base = cell_idx * cls.ELEMENTS_PER_CELL
            cell = dict(zip(cls.CELL_KEYS, values[base : base + cls.ELEMENTS_PER_CELL]))
            cell["cell_id"] = int(cell["cell_id"])
            cell["patch_id"] = int(cell["patch_id"])
            cell["segment_id"] = int(cell["segment_id"])
            cells.append(cell)
        return cells
