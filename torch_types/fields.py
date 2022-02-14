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
    @staticmethod
    @abstractmethod
    def collate_fn(batch: List["Field"]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, np.ndarray]:
        ...

    @abstractmethod
    def from_dict(self, data: Dict[str, np.ndarray]) -> "Field":
        ...


class ArrayField(Field):
    def __init__(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        super().__init__()

        value = _to_numpy(value)
        self.value = value
        self.shape = value.shape

    @staticmethod
    def collate_fn(batch: List["ArrayField"]) -> torch.Tensor:
        return torch.stack([torch.from_numpy(x.value) for x in batch], dim=0)


class ClassificationField(Field, Generic[IntEnumT]):
    Classes: Type[IntEnumT]

    def __init__(
        self,
        cls: Union[str, int, IntEnumT],
        logits: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        if isinstance(cls, int):
            cls = type(self).Classes(cls)
        elif isinstance(cls, str):
            cls = type(self).Classes[cls]
        self.cls = cls
        self.logits = logits

    @staticmethod
    def collate_fn(batch: List["ClassificationField"]) -> Dict[str, torch.Tensor]:
        res = {"cls": torch.tensor([b.cls for b in batch])}
        logits = [b.logits for b in batch]
        if all(l is not None for l in logits):
            res["logits"] = torch.tensor(logits)
        return res

    @classmethod
    def from_logits(cls, logits: np.ndarray) -> "ClassificationField":
        cls_ = cls.Classes(np.argmax(logits))
        return cls(cls_, logits)


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
        self.bboxes = bboxes

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
