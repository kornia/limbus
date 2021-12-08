"""Component definition."""
from abc import abstractmethod
from typing import Dict, Any, Collection, Optional
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


class Params:
    """Class to store parameters."""
    def __init__(self) -> None:
        self._types: Dict[str, Any] = {}
        self._args: Dict[str, Optional[str]] = {}

    def declare(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: Optional[str] = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, Union[str, int]...). Default: typing.Any
            value (optional): value for the parameter. Default: NoValue().
            arg (optional): argument directly related with the value of the parameter. Default: None.
                            E.g. this is useful to propagate datatypes and values from the target pin to the arg.

        """
        if not isinstance(value, NoValue):
            typeguard.check_type("value", value, tp)
        setattr(self, name, value)
        self._types[name] = tp
        self._args[name] = arg

    def get_related_arg(self, name: str) -> Optional[str]:
        """Return the argument related with a given param.

        Args:
            name: name of the param.

        """
        return self._args[name]

    def get_params(self) -> Collection[str]:
        """Return the name of all the params."""
        return self._types.keys()

    def get_types(self) -> Dict[str, type]:
        """Return the name and the type of all the params."""
        return self._types

    def get_type(self, name: str) -> type:
        """Return the type of a given param.

        Args:
            name: name of the param.

        """
        return self._types[name]

    def get_param(self, name: str) -> Any:
        """Return the param value after checking the type.

        Args:
            name: name of the param.

        """
        typeguard.check_type(name, getattr(self, name), self.get_type(name))
        return getattr(self, name)

    def set_param(self, name: str, value: Any) -> None:
        """Set the param value after checking the type.

        Args:
            name: name of the param.
            value: value to be setted.

        """
        typeguard.check_type(name, value, self.get_type(name))
        setattr(self, name, value)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __repr__(self) -> str:
        return ''.join(
            (
                f'{type(self).__name__}(',
                ', '.join(
                    f'{name}={getattr(self, name)}' for name in sorted(self.__dict__) if not name.startswith('_')
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
