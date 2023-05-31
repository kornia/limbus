"""Classes to define set of parameters."""
from __future__ import annotations
from typing import Any, Iterator, Iterable, Callable
from abc import ABC, abstractmethod

# Note that Component class cannot be imported to avoid circular dependencies.
# Since it is only used for type hints we import the module and use "component.Component" for typing.
from limbus.core import component
from limbus.core.param import Param, NoValue, InputParam, OutputParam, PropertyParam, InputEvent, OutputEvent, EventType


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


class InputEvents(Params):
    """Class to manage input events."""

    def declare(self, name: str, tp: Any = EventType, callback: Callable | None = None) -> None:
        """Add or modify an input event.

        Note: InputEvents do not need to be awaited but they can if the receiving component must be blocked.
              There are 4 types of events:
                - EventType: any event. You can use as you want: blocking/non blocking, exec a callback...
                - ComponentEventTypes defined in component.py. They are blocking events. They can also exec a callback.

        Args:
            name: name of the parameter.
            tp: type of the event. Default: EventType.
                Current valid types are:
                    - EventType: any event.
                    - ComponentEventTypes defined in component.py.
            callback: async callback function to be called when the event is sent. Note that it is executed before
                awaiting the event (if it is awaited).
                Prototype: `async def callback(parent: Component) -> None:`

        """
        if not issubclass(tp, EventType):
            raise TypeError(f"Invalid type for event {name}. It must be or inherit from EventType.")
        setattr(self, name, InputEvent(name, tp=tp, parent=self._parent, callback=callback))

    def __getattr__(self, name: str) -> InputEvent:  # type: ignore  # it should return an InitEvent
        """Trick to avoid mypy issues with dinamyc attributes."""
        ...


class OutputEvents(Params):
    """Class to manage output events."""

    def declare(self, name: str) -> None:
        """Add or modify an output event.

        Args:
            name: name of the parameter.

        """
        setattr(self, name, OutputEvent(name, tp=EventType, parent=self._parent))

    def __getattr__(self, name: str) -> OutputEvent:  # type: ignore  # it should return an OutputEvent
        """Trick to avoid mypy issues with dinamyc attributes."""
        ...
