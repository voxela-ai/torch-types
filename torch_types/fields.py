from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch

T = TypeVar("T")
IntEnumT = TypeVar("IntEnumT", bound=IntEnum)


def _to_numpy(x) -> np.ndarray:
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


class _NumberField(Generic[T], Field):
    Type: Type[T]
    value: T

    def __init__(self, value) -> None:
        self.value = type(self).Type(value)

    @staticmethod
    def collate_fn(batch: List["_NumberField"]) -> torch.Tensor:
        return torch.from_numpy(np.array([x.value for x in batch]))

    def to_dict(self) -> dict:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict):
        value = data["value"]
        return cls(value)


class Int64Field(_NumberField):
    Type = np.int64


class Float32Field(_NumberField):
    Type = np.float32


class ArrayField(Field):

    def __init__(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        super().__init__()

        value = _to_numpy(value)
        self.value = value
        self.shape = np.array(value.shape)
        self.dtype = np.dtype(value.dtype).name

    @staticmethod
    def collate_fn(batch: List["ArrayField"]) -> torch.Tensor:
        return torch.stack([torch.from_numpy(x.value.copy()) for x in batch], dim=0)

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
    """
    Stores normalized bounding boxes.
    """
    BBoxInputT = Union[None, np.ndarray, torch.Tensor]

    def __init__(
        self,
        /,
        shape: Tuple[int, int, int],
        xywh: BBoxInputT = None,
        xyxy: BBoxInputT = None,
    ) -> None:
        assert not (xywh is not None and xyxy is not None), "Specify only _one_ of xywh or xyxy"
        super().__init__()
        
        H, W, _ = shape
        if xywh is None and xyxy is None:
            bboxes = np.empty((0, 4))
        elif xyxy is not None:
            bboxes = _to_numpy(xyxy)
            bboxes = BBoxField.xyxy2xywh(xyxy)
        else:
            bboxes = _to_numpy(xywh)

        if bboxes.size == 0:
            bboxes = np.empty((0, 4))
        if bboxes.ndim == 1:
            bboxes = np.expand_dims(bboxes, 0)

        assert isinstance(bboxes, np.ndarray)
        assert bboxes.shape[-1] == 4
        self.xywhn = bboxes.astype(np.float32) / np.array([W, H, W, H])
        self.shape = np.array(shape).astype(np.int64)

    @property
    def xyxyn(self) -> np.ndarray:
        x, y, w, h = np.split(self.xywhn, 4, axis=-1)
        return np.concatenate((x, y, x + w, y + h), axis=-1)

    @property
    def xyxy(self) -> np.ndarray:
        x, y, w, h = np.split(self.xywh, 4, axis=-1)
        return np.concatenate((x, y, x + w, y + h), axis=-1)

    @property
    def xywh(self) -> np.ndarray:
        H, W, _ = self.shape
        return self.xywhn * np.array([W, H, W, H])

    @staticmethod
    def xyxy2xywh(xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = np.split(xyxy, 4, axis=-1)
        w = x2 - x1
        h = y2 - y1
        return np.concatenate((x1, y1, w, h), axis=-1)

    @staticmethod
    def collate_fn(batch: List["BBoxField"]) -> Dict[str, torch.Tensor]:
        bboxes = []
        batch_idx = []
        for i, b in enumerate(batch):
            bboxes.extend(torch.from_numpy(b.xywh))
            batch_idx.extend([torch.tensor(i)] * b.xywh.shape[0])

        if bboxes:
            bbox_tensor = torch.stack(bboxes, dim=0)
            idx_tensor = torch.stack(batch_idx, dim=0)
        else:
            bbox_tensor = torch.empty((0, 4))
            idx_tensor = torch.empty(0)

        return {
            "xywh": bbox_tensor,
            "batch_idx": idx_tensor,
        }

    def to_dict(self) -> dict:
        return {
            "xywh": self.xywh.flatten(),
            "nboxes": np.array([self.xywh.shape[0]], dtype=np.int64),
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BBoxField":
        bboxes, nboxes, shape = data["xywh"], data["nboxes"], data["shape"]
        assert isinstance(bboxes, np.ndarray)
        assert len(nboxes) == 1
        bboxes = bboxes.reshape(nboxes[0], 4)
        return cls(shape=shape, xywh=bboxes)


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
