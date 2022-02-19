from collections import defaultdict
from typing import (
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    OrderedDict,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch

from torch_types.fields import Field


class Instance(MutableMapping[str, Field]):

    __annotations__: OrderedDict[str, Type[Field]] = OrderedDict()
    field_types = __annotations__

    def __init_subclass__(cls, /, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        cls.field_types = cls.__annotations__
        for name, typ_ in cls.__annotations__.items():
            if not issubclass(typ_, Field):
                raise TypeError(f"Item is not a subclass of Field: {name} {typ_}")

    def __init__(self, **kwargs) -> None:
        self._data: Dict[str, Field] = dict(**kwargs)
        for name in self._data:
            if name not in self.field_types:
                raise ValueError(f"Name not a key in `{type(self)}`: {name}")

            typ_ = self.field_types[name]
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
        if key not in type(self).field_types:
            raise ValueError(f"Key is not a field of {type(self)}: {key}")
        typ_ = type(self).field_types[key]
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

    def __iter__(self) -> Iterator[str]:
        for key in type(self).field_types:
            if key in self._data:
                yield key

    def __len__(self) -> int:
        return len(self._data)

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

    def to_dict(self) -> dict:
        res = {}
        for key, val in self._data.items():
            res.update({f"{key}.{name}": v for name, v in val.to_dict().items()})
        return res

    @classmethod
    def from_dict(cls, data: dict) -> "Instance":
        res = defaultdict(dict)
        for key, val in data.items():
            name, attr_name = key.split(".")
            if name not in cls.field_types:
                raise ValueError(f"Key not in {cls}: {name}")
            res[name][attr_name] = val

        fields = {}
        for name in res:
            typ_ = cls.field_types[name]
            fields[name] = typ_.from_dict(res[name])
        return cls(**fields)
