"""Classes to define set of parameters."""
from __future__ import annotations
from typing import Any, Iterator, Iterable, Callable
from abc import ABC, abstractmethod

# Note that Component class cannot be imported to avoid circular dependencies.
# Since it is only used for type hints we import the module and use "component.Component" for typing.
from limbus.core import component
from limbus.core.param import Param, NoValue, InputParam, OutputParam, PropertyParam


class Params(Iterable, ABC):
    """Class to store parameters."""

    def __init__(self, parent_component: None | "component.Component" = None):
        super().__init__()
        self._parent = parent_component

    @abstractmethod
    def declare(self, *args, **kwargs) -> None:
        """Add or modify a param."""
        raise NotImplementedError

    def get_params(self, only_connected: bool = False) -> list[str]:
        """Return the name of all the params.

        Args:
            only_connected: If True, only return the params that are connected.

        """
        params = []
        for name in self.__dict__:
            param = getattr(self, name)
            if isinstance(param, Param) and (not only_connected or param.ref_counter()):
                params.append(name)
        return params

    def __len__(self) -> int:
        return len(self.get_params())

    def __getitem__(self, name: str) -> Param:
        return getattr(self, name)

    def __iter__(self) -> Iterator[Param]:
        for name in self.__dict__:
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


class InputParams(Params):
    """Class to manage input parameters."""

    def declare(self, name: str, tp: Any = Any, value: Any = NoValue(), callback: Callable | None = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, str | int,...). Default: typing.Any
            value (optional): value for the parameter. Default: NoValue().
            callback (optional): async callback function to be called when the parameter value changes.
                Prototype: `async def callback(parent: Component, value: TYPE) -> TYPE:`
                    - MUST return the value to be finally used.
                Default: None.

        """
        if isinstance(value, Param):
            value = value.value
        setattr(self, name, InputParam(name, tp, value, None, self._parent, callback))

    def __getattr__(self, name: str) -> InputParam:  # type: ignore  # it should return an InitParam
        """Trick to avoid mypy issues with dinamyc attributes."""
        ...


class PropertyParams(Params):
    """Class to manage property parameters."""

    def declare(self, name: str, tp: Any = Any, value: Any = NoValue(), callback: Callable | None = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, str | int,...). Default: typing.Any
            value (optional): value for the parameter. Default: NoValue().
            callback (optional): async callback function to be called when the parameter value changes.
                Prototype: `async def callback(parent: Component, value: TYPE) -> TYPE:`
                    - MUST return the value to be finally used.
                Default: None.

        """
        if isinstance(value, Param):
            value = value.value
        setattr(self, name, PropertyParam(name, tp, value, None, self._parent, callback))

    def __getattr__(self, name: str) -> PropertyParam:  # type: ignore  # it should return an PropParam
        """Trick to avoid mypy issues with dinamyc attributes."""
        ...


class OutputParams(Params):
    """Class to manage output parameters."""

    def declare(self, name: str, tp: Any = Any, arg: None | str = None, callback: Callable | None = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, str | int,...). Default: typing.Any
            arg (optional): Component argument directly related with the value of the parameter. Default: None.
                E.g. this is useful to propagate datatypes and values from a pin with a default value to an argument
                in a Component (GUI).
            callback (optional): async callback function to be called when the parameter value changes.
                Prototype: `async def callback(parent: Component, value: TYPE) -> TYPE:`
                    - MUST return the value to be finally used.
                Default: None.

        """
        setattr(self, name, OutputParam(name, tp, NoValue(), arg, self._parent, callback))

    def __getattr__(self, name: str) -> OutputParam:  # type: ignore  # it should return an OutputParam
        """Trick to avoid mypy issues with dinamyc attributes."""
        ...
