"""Classes to define set of parameters."""
from __future__ import annotations
from typing import Any, Iterator, Iterable, Callable

# Note that Component class cannot be imported to avoid circular dependencies.
# Since it is only used for type hints we import the module and use "component.Component" for typing.
from limbus.core import component
from limbus.core.param import Param, NoValue, InputParam, OutputParam


class Params(Iterable):
    """Class to store parameters."""

    def __init__(self, parent_component: None | "component.Component" = None):
        super().__init__()
        self._parent = parent_component

    def declare(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: None | str = None,
                callback: Callable | None = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, str | int,...). Default: typing.Any
            value (optional): value for the parameter. Default: NoValue().
            arg (optional): Component argument directly related with the value of the parameter. Default: None.
                            E.g. this is useful to propagate datatypes and values from a pin with a default value to
                            an argument in a Component (GUI).
            callback (optional): callback function to be called when the parameter value changes. Default: None.

        """
        if isinstance(value, Param):
            value = value.value
        setattr(self, name, Param(name, tp, value, arg, self._parent, callback))

    def __getattr__(self, name: str) -> Param:  # type: ignore  # it should return a Param
        """Trick to avoid mypy issues with dinamyc attributes."""
        ...

    def get_related_arg(self, name: str) -> None | str:
        """Return the argument in the Component constructor related with a given param.

        Args:
            name: name of the param.

        """
        return getattr(self, name).arg

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

    def get_types(self) -> dict[str, type]:
        """Return the name and the type of all the params."""
        types: dict[str, type] = {
            name: getattr(self, name).type for name in self.__dict__ if not name.startswith('_')}
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

    def declare(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: None | str = None,
                callback: Callable | None = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, str | int,...). Default: typing.Any
            value (optional): value for the parameter. Default: NoValue().
            arg (optional): Component argument directly related with the value of the parameter. Default: None.
                            E.g. this is useful to propagate datatypes and values from a pin with a default value to
                            an argument in a Component (GUI).
            callback (optional): callback function to be called when the parameter value changes. Default: None.

        """
        if isinstance(value, Param):
            value = value.value
        setattr(self, name, InputParam(name, tp, value, arg, self._parent, callback))

    def __getattr__(self, name: str) -> InputParam:  # type: ignore  # it should return an InitParam
        """Trick to avoid mypy issues with dinamyc attributes."""
        ...


class OutputParams(Params):
    """Class to manage output parameters."""

    def declare(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: None | str = None,
                callback: Callable | None = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, str | int,...). Default: typing.Any
            value (optional): value for the parameter. Default: NoValue().
            arg (optional): Component argument directly related with the value of the parameter. Default: None.
                            E.g. this is useful to propagate datatypes and values from a pin with a default value to
                            an argument in a Component (GUI).
            callback (optional): callback function to be called when the parameter value changes. Default: None.

        """
        if isinstance(value, Param):
            value = value.value
        setattr(self, name, OutputParam(name, tp, value, arg, self._parent, callback))

    def __getattr__(self, name: str) -> OutputParam:  # type: ignore  # it should return an OutputParam
        """Trick to avoid mypy issues with dinamyc attributes."""
        ...
