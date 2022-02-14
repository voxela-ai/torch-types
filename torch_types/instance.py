from collections import defaultdict
from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    MutableMapping,
    Optional,
    OrderedDict,
    Tuple,
    Type,
)

import torch

from torch_types.fields import Field


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
