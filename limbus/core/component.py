"""Component definition."""
from dataclasses import dataclass
from abc import abstractmethod
from typing import Dict, Any, Collection, Optional, List
from enum import Enum

import typeguard
import torch.nn as nn


class ComponentState(Enum):
    """Possible states for the components."""
    STOPPED = 0
    OK = 1
    NotImplemented = -1
    ERROR = 2
    DISABLED = 3


class NoValue():
    """Denote that a param does not have a value."""
    pass


@dataclass
class Value():
    """Denote that a param has a value."""
    value: Any


class Param:
    """Class to store data for each parameter."""
    def __init__(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: Optional[str] = None) -> None:
        self._name: str = name
        self._value: Value = Value(value)
        self._type: Any = tp
        self._arg: Optional[str] = arg
        self._refs: List["Param"] = []
        # validate that the type is coherent with the value
        if not isinstance(value, NoValue):
            typeguard.check_type(name, value, tp)

    @property
    def arg(self) -> Optional[str]:
        """Get the argument realted with the param."""
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
    def value(self) -> Any:
        """Get the value of the parameter."""
        return self._value.value

    @value.setter
    def value(self, value: Any) -> None:
        """Set the value of the parameter.

        Args:
            value (Any): The value to set.

        """
        if isinstance(value, Param):
            value = value.value
        typeguard.check_type(self._name, value, self._type)
        self._value.value = value

    def connect(self, dst: "Param") -> None:
        """Connect this parameter with the dst parameter."""
        # TODO: check that dst param is an input param
        # TODO: check type compatibility
        dst._value = self._value
        self._refs.append(dst)
        dst._refs.append(self)
        if len(dst._refs) > 1:
            raise ValueError(f"An input parameter can only be connected to 1 param. "
                             f"Dst param '{dst.name}' is connected to {dst._refs}.")

    def disconnect(self, dst: "Param") -> None:
        """Disconnect this parameter from the dst parameter."""
        try:
            self._refs.remove(dst)
        except:
            pass
        try:
            dst._refs.remove(self)
        except:
            pass
        if len(dst._refs) == 0:
            dst._value = Value(self._value.value)
        else:
            raise ValueError(f"An input parameter can only be connected to 1 param. "
                             f"Dst param '{dst.name}' is connected to {dst._refs}.")


class Params:
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

    def get_params(self) -> Collection[str]:
        """Return the name of all the params."""
        return {name for name in sorted(self.__dict__) if not name.startswith('_')}

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

    def set_param(self, name: str, value: Any) -> None:
        """Set the param value.

        Args:
            name: name of the param.
            value: value to be setted.

        """
        getattr(self, name).value = value

    def __getitem__(self, name: str) -> Param:
        return getattr(self, name)

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
    def forward(self, inputs: Params) -> ComponentState:
        """Run the component.

        Args:
         inputs: set of values to be used to run the component.

        """
        return ComponentState.NotImplemented

    def finish_iter(self) -> None:
        """Event executed when a pipeline iter is finished."""
        pass
