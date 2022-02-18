from abc import ABCMeta, abstractmethod
from typing import Generic, Type, TypeVar

from torch_types.instance import Instance

InstanceT = TypeVar("InstanceT", bound=Instance)


class BaseWriter(metaclass=ABCMeta):
    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def write(self, instance: Instance) -> None:
        ...


class BaseReader(Generic[InstanceT], metaclass=ABCMeta):
    def __init__(self, instance_type: Type[InstanceT]) -> None:
        super().__init__()
        self.instance_type = instance_type

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self) -> InstanceT:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
