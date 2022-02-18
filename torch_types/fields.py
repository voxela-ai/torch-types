from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import Dict, Generic, List, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import torch

IntEnumT = TypeVar("IntEnumT", bound=IntEnum)


def _to_numpy(x: Union[Sequence, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


class Field(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def collate_fn(batch: List["Field"]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "Field":
        ...


class ArrayField(Field):
    value: np.ndarray
    shape: np.ndarray
    dtype: str

    def __init__(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        super().__init__()

        value = _to_numpy(value)
        self.value = value
        self.shape = np.array(value.shape)
        self.dtype = np.dtype(value.dtype).name

    @staticmethod
    def collate_fn(batch: List["ArrayField"]) -> torch.Tensor:
        return torch.stack([torch.from_numpy(x.value) for x in batch], dim=0)

    def to_dict(self) -> dict:
        return {
            "value": self.value.tobytes(),
            "shape": self.shape,
            "dtype": bytes(self.dtype, "utf8"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArrayField":
        value, shape, dtype = data["value"], data["shape"], data["dtype"]
        assert isinstance(value, bytes)
        assert isinstance(shape, np.ndarray)
        assert isinstance(dtype, bytes)
        return cls(np.frombuffer(value, dtype=dtype.decode()).reshape(shape))


class ClassificationField(Field, Generic[IntEnumT]):
    Classes: Type[IntEnumT]

    def __init__(
        self,
        clss: Union[str, int, IntEnumT],
        logits: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        if isinstance(clss, int):
            clss = type(self).Classes(clss)
        elif isinstance(clss, str):
            clss = type(self).Classes[clss]
        self.clss = clss
        self.logits = logits

    @staticmethod
    def collate_fn(batch: List["ClassificationField"]) -> Dict[str, torch.Tensor]:
        res = {"clss": torch.tensor([b.clss for b in batch])}
        logits = [b.logits for b in batch]
        if all(l is not None for l in logits):
            res["logits"] = torch.tensor(logits)
        return res

    def to_dict(self) -> dict:
        res = {"clss": np.array([self.clss], dtype=np.int64)}
        if self.logits:
            res["logits"] = self.logits
        return res

    @classmethod
    def from_dict(cls, data: dict) -> "ClassificationField":
        assert isinstance(data["clss"], np.ndarray)
        clss = data["clss"].item()
        logits = data.get("logits", None)
        return cls(clss=clss, logits=logits)

    @classmethod
    def from_logits(cls, logits: np.ndarray) -> "ClassificationField":
        clss = cls.Classes(np.argmax(logits))
        return cls(clss, logits)


class BBoxField(Field):
    def __init__(self, bboxes: Union[None, np.ndarray, torch.Tensor]) -> None:
        super().__init__()
        if bboxes is None:
            bboxes = np.empty((0, 4))
        bboxes = _to_numpy(bboxes)
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, 0)

        assert isinstance(bboxes, np.ndarray)
        assert bboxes.shape[-1] == 4
        self.bboxes = bboxes.astype(np.float32)

    @staticmethod
    def collate_fn(batch: List["BBoxField"]) -> Dict[str, torch.Tensor]:
        bboxes = []
        batch_idx = []
        for i, b in enumerate(batch):
            bboxes.extend(torch.from_numpy(b.bboxes))
            batch_idx.extend([i] * b.bboxes.shape[0])

        if bboxes:
            bbox_tensor = torch.stack(bboxes, dim=0)
            idx_tensor = torch.stack(batch_idx, dim=0)
        else:
            bbox_tensor = torch.empty((0, 4))
            idx_tensor = torch.empty((0, 4))

        return {
            "bboxes": bbox_tensor,
            "batch_idx": idx_tensor,
        }

    def to_dict(self) -> dict:
        return {
            "bboxes": self.bboxes.flatten(),
            "nboxes": np.array([self.bboxes.shape[0]], dtype=np.int64),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BBoxField":
        bboxes, nboxes = data["bboxes"], data["nboxes"]
        assert isinstance(bboxes, np.ndarray)
        assert len(nboxes) == 1
        bboxes = bboxes.reshape(nboxes[0], 4)
        return cls(bboxes=bboxes)


class StringField(Field):
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    @staticmethod
    def collate_fn(batch: List["StringField"]) -> List[str]:
        return [b.value for b in batch]

    def to_dict(self) -> dict:
        return {"value": self.value.encode("utf-8")}

    @classmethod
    def from_dict(cls, data: dict) -> "StringField":
        return cls(data["value"].decode("utf-8"))
