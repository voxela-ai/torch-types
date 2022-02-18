import struct
from pathlib import Path
from typing import List, Tuple, Type, TypeVar

import numpy as np
from tfrecord import example_pb2
from tfrecord.writer import TFRecordWriter as _TFRecordWriter

from .base import BaseReader, BaseWriter
from torch_types.instance import Instance

INDEX_SUFFIX = ".index"
InstanceT = TypeVar("InstanceT", bound=Instance)


class TFRecordWriter(_TFRecordWriter, BaseWriter):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)
        self.index_file = Path(data_path).with_suffix(INDEX_SUFFIX).open("w")

    def close(self) -> None:
        super().close()
        self.index_file.close()

    def write(self, instance: Instance) -> None:
        start = self.file.tell()
        data = instance.to_dict()
        super().write(TFRecordWriter._attach_types(data))
        length = self.file.tell() - start
        self.index_file.write(f"{start} {length}\n")

    @staticmethod
    def _attach_types(data: dict) -> dict:
        res = {}
        for key in data:
            val = data[key]
            if isinstance(val, bytes):
                dtype_str = "byte"
            elif isinstance(val, np.ndarray):
                if val.dtype is np.dtype("<f4"):
                    dtype_str = "float"
                elif val.dtype is np.dtype("<i8"):
                    dtype_str = "int"
                else:
                    raise ValueError(f"Unsupported dtype {val.dtype} for key {key}")
            else:
                raise ValueError(f"Unsupported type {type(val)} for key {key}")
            res[key] = (val, dtype_str)
        return res
            


class TFRecordReader(BaseReader[InstanceT]):
    _tftype_convert_map = {
        "bytes_list": lambda x: x[0],
        "float_list": lambda x: np.array(x, dtype=np.float32),
        "int64_list": lambda x: np.array(x, dtype=np.int64),
    }

    def __init__(self, data_path: str, instance_type: Type[InstanceT]) -> None:
        super().__init__(instance_type)

        path = Path(data_path)
        self.data_file = path.with_suffix(".tfrecord").open("rb")
        self.instance_type = instance_type
        self.offsets: List[Tuple[int, int]] = []
        with path.with_suffix(".index").open() as f:
            for line in f.readlines():
                start, length = line.split()
                self.offsets.append((int(start), int(length)))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> InstanceT:
        offset, length = self.offsets[idx]
        example = self.read_example(offset)
        if self.data_file.tell() - offset != length:
            raise RuntimeError("Example does not match index length")
        return self.features_to_instance(example.features)

    def close(self) -> None:
        self.data_file.close()

    def features_to_instance(self, features: example_pb2.Features) -> InstanceT:
        feature = features.feature
        data = {}
        # get only the fields that are in the Instance type
        for f_name, f_val in feature.items():
            field_name = f_name.split(".")[0]
            if field_name in self.instance_type.field_types:
                desc, val_list = f_val.ListFields()[0]
                fn = TFRecordReader._tftype_convert_map[desc.name]
                data[f_name] = fn(val_list.value)
        return self.instance_type.from_dict(data)

    def read_example(self, offset: int) -> example_pb2.Example:
        if self.data_file.tell() != offset:
            self.data_file.seek(offset)

        header_buffer = bytearray(8)
        crc_buffer = bytearray(4)
        data_buffer = bytearray(1024 * 1024)

        if self.data_file.readinto(header_buffer) != 8:
            raise RuntimeError("Failed to read the record size.")
        if self.data_file.readinto(crc_buffer) != 4:
            raise RuntimeError("Failed to read the start token.")

        (length,) = struct.unpack("<Q", header_buffer)
        if length > len(data_buffer):
            data_buffer = data_buffer.zfill(int(length * 1.5))
        data_bytes_view = memoryview(data_buffer)[:length]

        if self.data_file.readinto(data_bytes_view) != length:
            raise RuntimeError("Failed to read the record.")
        if self.data_file.readinto(crc_buffer) != 4:
            raise RuntimeError("Failed to read the end token.")

        example = example_pb2.Example()
        example.ParseFromString(data_bytes_view)  # type: ignore
        return example
