from abc import ABCMeta, abstractmethod
from collections import defaultdict
from enum import IntEnum
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    MutableMapping,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

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


class ArrayField(Field):
    def __init__(self, value: Union[np.ndarray, torch.Tensor]) -> None:
        super().__init__()

        value = _to_numpy(value)
        self.value = value

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


class Instance(MutableMapping[str, Field]):

    __annotations__: OrderedDict[str, Type[Field]] = OrderedDict()
    _field_types = __annotations__

    def __init_subclass__(cls, /, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        for name, typ_ in cls.__annotations__.items():
            if not issubclass(typ_, Field):
                raise TypeError(f"Item is not a subclass of Field: {name} {typ_}")

    def __init__(self, **kwargs) -> None:
        self._data: Dict[str, Field] = dict(**kwargs)
        for name in self._data:
            if name not in self._field_types:
                raise ValueError(f"Name not a key in `{type(self)}`: {name}")

            typ_ = self._field_types[name]
            val = self._data[name]
            if not isinstance(val, typ_):
                if isinstance(val, dict):
                    self._data[name] = typ_(**val)
                else:
                    self._data[name] = typ_(val)

    def __getitem__(self, key: str) -> Field:
        return self._data[key]

    def __getattr__(self, key: str) -> Field:
        return self.__getitem__(key)

    def __setitem__(self, key: str, value: Field) -> None:
        if key not in type(self)._field_types:
            raise ValueError(f"Key is not a field of {type(self)}: {key}")
        typ_ = type(self)._field_types[key]
        if not isinstance(value, typ_):
            raise ValueError(f"Value is not a {typ_}: {type(value)}")
        self._data[key] = value

    def __setattr__(self, key: str, value: Field) -> None:
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[Optional[Field]]:
        for key in type(self)._field_types:
            yield self._data.get(key, None)

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> Iterator[Tuple[str, Field]]:
        for key in type(self)._field_types:
            if key in self._data:
                yield key, self._data[key]

    @classmethod
    def collate_fn(cls, batch: List["Instance"]) -> Dict[str, torch.Tensor]:
        batch_map = defaultdict(list)
        type_map: Dict[str, Type[Field]] = {}
        for b in batch:
            for key, val in b.items():
                typ_ = type(val)
                batch_map[key].append(val)
                if key in type_map:
                    assert typ_ is type_map[key]
                else:
                    type_map[key] = typ_

        res: Dict[str, torch.Tensor] = {}
        for key, val in batch_map.items():
            typ_ = type_map[key]
            collated_field = typ_.collate_fn(val)
            if isinstance(collated_field, torch.Tensor):
                res[key] = collated_field
            elif isinstance(collated_field, dict):
                for sub_key, sub_val in collated_field.items():
                    res[f"{key}.{sub_key}"] = sub_val
        return res
