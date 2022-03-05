"""Component definition."""
from dataclasses import dataclass
from abc import abstractmethod
from collections import defaultdict
import typing
from typing import Dict, Any, Set, Optional, List, Iterator, Iterable, Union, Tuple
from enum import Enum
import inspect
import collections

import typeguard
import torch.nn as nn
import numpy as np


class ComponentState(Enum):
    """Possible states for the components."""
    STOPPED = 0
    OK = 1
    NotImplemented = -1
    ERROR = 2
    DISABLED = 3


class NoValue:
    """Denote that a param does not have a value."""
    pass


@dataclass
class Container:
    """Denote that a param has a value."""
    value: Any


@dataclass
class IterableContainer:
    """Denote that a param has an indexed value.

    Note: In our use case the maximum number of nested IterableContainers is 2.
    This number is not explicitly controlled. It is implicitly controlled in the Param class.

    """
    container: Union[Container, "IterableContainer"]
    index: int

    @property
    def value(self) -> Any:
        """Get the value of the container."""
        if isinstance(self.container, Container):
            # return the value of the container at the index
            return self.container.value[self.index]
        else:
            # look for the container value.
            # If it is an IterableContainer means that the final value is nested.
            assert isinstance(self.container, IterableContainer)
            return self.container.value


class IterableInputContainers:
    """Denote that an input param is a sequence of Containers."""
    def __init__(self, container: Optional[IterableContainer] = None):
        containers = []
        if container is not None:
            containers = [container]
        self._containers: List[IterableContainer] = containers

    def __len__(self) -> int:
        return len(self._containers)

    def add(self, container: IterableContainer) -> None:
        """Add an IterableValue to the list of values."""
        self._containers.append(container)

    def remove(self, index: int) -> None:
        """Remove an IterableValue from the list of values."""
        for container in self._containers:
            if container.index == index:
                self._containers.remove(container)
                return

    def get_ordered(self) -> List[Any]:
        """Return a list with the values in the order denoted by the index in the IterableValue."""
        indices: List[int] = []
        for container in self._containers:
            assert isinstance(container, IterableContainer)
            indices.append(container.index)

        containers: List[Any] = []
        for pos_idx in np.argsort(indices):
            # we asume cannot be empty elements, so we can append all the values
            obj: Union[Container, IterableContainer] = self._containers[pos_idx].container
            if isinstance(obj, IterableContainer):
                obj = obj.container.value[obj.index]  # type: ignore  # Iterable[Any] is not indexable [index]
            else:
                assert isinstance(obj, Container)
                obj = obj.value
            containers.append(obj)
        return containers


def _check_subscriptable(datatype: type) -> bool:
    """Checf if datatype is subscriptable.

    Args:
        datatype (type): type to be analised.

    Returns:
        bool: True if datatype is subscriptable, False otherwise.

    """
    if inspect.isclass(datatype):
        return False

    origin = typing.get_origin(datatype)
    if inspect.isabstract(origin) and (collections.abc.Sequence or collections.abc.Iterable):
        return True
    # mypy complaints in the case origin is NoneType
    if (inspect.isclass(origin) and isinstance(origin(), typing.Iterable)):  # type: ignore
        return True
    return False


class IterableParam:
    """Temporal class to manage indexing inside a parameter."""
    def __init__(self, param: "Param", index: int) -> None:
        self._param: Param = param
        self._index: int = index
        self._iter_value: Union[IterableContainer, IterableInputContainers]
        if isinstance(param._value, Container):
            self._iter_value = IterableContainer(param._value, index)
        elif isinstance(param._value, IterableInputContainers):
            # since it is an input, the pointer to the value is not relevant at this stage
            self._iter_value = IterableContainer(Container(None), index)

    @property
    def param(self) -> "Param":
        """Return the base parameter."""
        return self._param

    @property
    def index(self) -> int:
        """Return the selected index in the sequence."""
        return self._index

    @property
    def value(self) -> Union[IterableContainer, IterableInputContainers]:
        """Get the value of the parameter."""
        return self._iter_value

    def ref_counter(self) -> int:
        """Return the number of references for this parameter."""
        return self._param.ref_counter(self._index)

    def connect(self, dst: Union["Param", "IterableParam"]) -> None:
        """Connect this parameter with the dst parameter."""
        self._param._connect(self, dst)

    def disconnect(self, dst: Union["Param", "IterableParam"]) -> None:
        """Connect this parameter with the dst parameter."""
        self._param._disconnect(self, dst)


class Param:
    """Class to store data for each parameter."""
    def __init__(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: Optional[str] = None) -> None:
        # validate that the type is coherent with the value
        if not isinstance(value, NoValue):
            typeguard.check_type(name, value, tp)

        self._name: str = name
        self._type: Any = tp
        self._arg: Optional[str] = arg
        self._refs: Dict[Any, Set[Tuple["Param", Optional[int]]]] = defaultdict(set)
        self._value: Union[Container, IterableContainer, IterableInputContainers] = Container(value)
        self._is_subscriptable = _check_subscriptable(tp)

    @property
    def arg(self) -> Optional[str]:
        """Get the argument related with the param."""
        return self._arg

    @property
    def type(self) -> Any:
        """Return the type of the parameter."""
        return self._type

    @property
    def name(self) -> str:
        """Get the name of the parameter."""
        return self._name

    @property
    def references(self) -> Set[Tuple["Param", Optional[int]]]:
        """Get all the references for the parameter."""
        refs: Set[Tuple["Param", Optional[int]]] = set()
        for ref_set in self._refs.values():
            refs = refs.union(ref_set)
        return refs

    @property
    def value(self) -> Any:
        """Get the value of the parameter."""
        if isinstance(self._value, Container):
            if isinstance(self._value.value, IterableContainer):
                # mypy error: Iterable[Any] is not indexable [index]
                return self._value.value.container.value[self._value.value.index]  # type: ignore
            else:
                return self._value.value
            return self._value.value
        elif isinstance(self._value, IterableInputContainers):
            assert self._is_subscriptable
            origin = typing.get_origin(self._type)
            assert origin is not None
            res_value: List[Any] = self._value.get_ordered()
            return origin(res_value)

    @value.setter
    def value(self, value: Any) -> None:
        """Set the value of the parameter.

        Args:
            value (Any): The value to set.

        """
        if isinstance(value, Param):
            value = value.value
        if not isinstance(self._value, Container):
            raise TypeError(f"Param '{self.name}' cannot be assigned.")
        if isinstance(value, (Container, IterableContainer, Set)):
            raise TypeError(f"The type of the value to be assigned to param '{self.name}' cannot have a Value type.")
        typeguard.check_type(self._name, value, self._type)
        self._value.value = value

    @property
    def container(self) -> Union[Container, IterableContainer, IterableInputContainers]:
        """Get the container for this parameter."""
        return self._value

    @container.setter
    def container(self, value: Union[Container, IterableContainer, IterableInputContainers]) -> None:
        """Set the container for this parameter.

        Args:
            value (Container, IterableContainer or IterableInputContainers): The container to set.

        """
        self._value = value

    def ref_counter(self, index: Optional[int] = None) -> int:
        """Return the number of references for this parameter."""
        if index is not None:
            return len(self._refs[index])
        else:
            return len(self.references)

    def select(self, index: int) -> IterableParam:
        """Select a slice of the parameter.

        Args:
            index (int): The index of the slice.

        Returns:
            Param: The selected slice.

        """
        if not self._is_subscriptable:
            raise ValueError(f"The param '{self.name}' is not a subscriptable.")
        # NOTE: we cannot check if the index is valid because it is not known at this point the len of the sequence
        # create a new param with the selected slice inside the param
        return IterableParam(self, index)

    def _connect(self, ori: Union["Param", IterableParam], dst: Union["Param", IterableParam]) -> None:
        """Connect this parameter with the dst parameter."""
        if self._is_subscriptable and isinstance(ori, Param):
            raise ValueError(f"The param '{self.name}' must be connected using indexes.")

        # TODO: check that dst param is an input param
        # TODO: check type compatibility
        if (isinstance(dst, Param) and dst.ref_counter() > 0):
            raise ValueError(f"An input parameter can only be connected to 1 param. "
                             f"Dst param '{dst.name}' is connected to {dst._refs}.")

        if isinstance(dst, IterableParam) and dst.param.ref_counter(dst.index) > 0:
            raise ValueError(f"An input parameter can only be connected to 1 param. "
                             f"Dst param '{dst.param.name}' is connected to {dst.param._refs}.")

        # connect the param to the dst param
        if isinstance(dst, Param) and isinstance(ori, Param):
            assert isinstance(dst._value, Container)
            assert isinstance(ori._value, Container)
            dst._value = ori._value
        elif isinstance(dst, IterableParam) and isinstance(ori, Param):
            assert isinstance(dst._iter_value, IterableContainer)
            assert isinstance(ori._value, Container)
            dst._iter_value.container = ori._value
        elif isinstance(dst, Param) and isinstance(ori, IterableParam):
            assert isinstance(dst._value, Container)
            assert isinstance(ori._iter_value, IterableContainer)
            dst._value.value = ori._iter_value
        else:
            assert isinstance(dst, IterableParam)
            assert isinstance(ori, IterableParam)
            assert isinstance(dst._iter_value, IterableContainer)
            assert isinstance(ori._iter_value, IterableContainer)
            dst._iter_value.container = ori._iter_value

        # if dest is an IterableParam means that several ori params can be connected to different dest indexes
        # so they should be saved as a set of params
        if isinstance(dst, IterableParam):
            assert isinstance(dst._iter_value, IterableContainer)
            if isinstance(dst.param._value, IterableInputContainers):
                dst.param._value.add(dst._iter_value)
            else:
                dst.param._value = IterableInputContainers(dst._iter_value)

        self._update_references('add', ori, dst)

    def connect(self, dst: Union["Param", IterableParam]) -> None:
        """Connect this parameter with the dst parameter."""
        self._connect(self, dst)

    def _disconnect(self, ori: Union["Param", IterableParam], dst: Union["Param", IterableParam]) -> None:
        """Disconnect this parameter from the dst parameter."""
        if isinstance(dst, Param):
            assert isinstance(dst._value, Container)
            dst._value = Container(NoValue())
        elif isinstance(dst, IterableParam):
            if isinstance(dst.param._value, IterableInputContainers):
                assert isinstance(dst._iter_value, IterableContainer)
                dst.param._value.remove(dst._iter_value.index)
                if len(dst.param._value) == 0:
                    dst.param._value = Container(NoValue())
            else:
                dst.param._value = Container(NoValue())

        self._update_references('remove', ori, dst)

    def _update_references(self, type: str, ori: Union["Param", IterableParam], dst: Union["Param", IterableParam]
                           ) -> None:
        # assign references
        ori_idx = None
        dst_idx = None
        if isinstance(ori, IterableParam):
            ori_idx = ori.index
            ori = ori.param
        if isinstance(dst, IterableParam):
            dst_idx = dst.index
            dst = dst.param
        if type == 'add':
            ori._refs[ori_idx].add((dst, dst_idx))
            dst._refs[dst_idx].add((ori, ori_idx))
        elif type == 'remove':
            ori._refs[ori_idx].remove((dst, dst_idx))
            dst._refs[dst_idx].remove((ori, ori_idx))

    def disconnect(self, dst: Union["Param", IterableParam]) -> None:
        """Disconnect this parameter from the dst parameter."""
        self._disconnect(self, dst)


class Params(Iterable):
    """Class to store parameters."""

    def declare(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: Optional[str] = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, Union[str, int]...). Default: typing.Any
            value (optional): value for the parameter. Default: NoValue().
            arg (optional): Component argument directly related with the value of the parameter. Default: None.
                            E.g. this is useful to propagate datatypes and values from a pin with a default value to
                            an argument in a Component (GUI).

        """
        if isinstance(value, Param):
            value = value.value
        setattr(self, name, Param(name, tp, value, arg))

    def get_related_arg(self, name: str) -> Optional[str]:
        """Return the Component argument related with a given param.

        Args:
            name: name of the param.

        """
        return getattr(self, name).arg

    def get_params(self, only_connected: bool = False) -> Set[str]:
        """Return the name of all the params.

        Args:
            only_connected: If True, only return the params that are connected.

        """
        params = set()
        for name in sorted(self.__dict__):
            param = getattr(self, name)
            if isinstance(param, Param) and (not only_connected or param.ref_counter()):
                params.add(name)
        return params

    def get_types(self) -> Dict[str, type]:
        """Return the name and the type of all the params."""
        types: Dict[str, type] = {
            name: getattr(self, name).type for name in sorted(self.__dict__) if not name.startswith('_')}
        return types

    def get_type(self, name: str) -> type:
        """Return the type of a given param.

        Args:
            name: name of the param.

        """
        return getattr(self, name).type

    def get_param(self, name: str) -> Any:
        """Return the param value.

        Args:
            name: name of the param.

        """
        return getattr(self, name).value

    def __len__(self) -> int:
        return len(self.get_params())

    def set_param(self, name: str, value: Any) -> None:
        """Set the param value.

        Args:
            name: name of the param.
            value: value to be setted.

        """
        getattr(self, name).value = value

    def __getitem__(self, name: str) -> Param:
        return getattr(self, name)

    def __iter__(self) -> Iterator[Param]:
        for name in sorted(self.__dict__):
            attr = getattr(self, name)
            if isinstance(attr, Param):
                yield attr

    def __repr__(self) -> str:
        return ''.join(
            (
                f'{type(self).__name__}(',
                ', '.join(
                    f'{name}={getattr(self, name).value}' for name in sorted(self.__dict__) if not name.startswith('_')
                ),
                ')',
            )
        )


class Component(nn.Module):
    """Base class to define a Limbus Component."""
    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._inputs = self.__class__.register_inputs()
        self._outputs = self.__class__.register_outputs()

    @staticmethod
    def register_inputs() -> Params:
        """Register the input params.

        Returns:
            input params

        """
        return Params()

    @staticmethod
    def register_outputs() -> Params:
        """Register the output params.

        Returns:
            output params

        """
        return Params()

    @property
    def name(self) -> str:
        """Name of the component."""
        return self._name

    @property
    def inputs(self) -> Params:
        """Get the set of component inputs."""
        return self._inputs

    @property
    def outputs(self) -> Params:
        """Get the set of component outputs."""
        return self._outputs

    @abstractmethod
    def forward(self) -> ComponentState:
        """Run the component."""
        return ComponentState.NotImplemented

    def finish_iter(self) -> None:
        """Event executed when a pipeline iter is finished."""
        pass
