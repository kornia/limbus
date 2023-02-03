"""Module to manage the visualization tools in limbus."""
from typing import Optional, Type, Union
import inspect

from limbus.widgets import types


# global var to store the visualization backend. We want a single instance.
_viz: Optional[types.Viz] = None
# global var to store the type used for the visualization backend.
_viz_cls: Type[types.Viz] = types.Console  # Default value is Console.


def set_type(viz_cls: Union[Type[types.Viz], str]) -> None:
    """Set the visualization class that will be used to create the visualization object.

    Args:
        viz_cls: The visualization class that will be used to create the visualization object.
            It can be a string with the name of the class (defined in limbus/viz/types.py) or the class itself.

    """
    global _viz_cls
    if isinstance(viz_cls, str):
        # the param viz_cls must be the name of the class.
        # So, we go through all the posible classes and find the one with the name.
        found = False
        for cls in inspect.getmembers(types, inspect.isclass):
            if (issubclass(cls[1], types.Viz) and
                    cls[1].__name__ != types.Viz.__name__ and
                    cls[0].lower() == viz_cls.lower()):
                _viz_cls = cls[1]
                found = True
                break
        if not found:
            raise ValueError(f"Unknown visualization type: {viz_cls}.")
    elif issubclass(viz_cls, types.Viz):
        if viz_cls == types.Viz:
            raise ValueError(f"Invalid visualization type. The Viz base class cannot be setted.")
        _viz_cls = viz_cls
    else:
        raise ValueError(f"Invalid visualization type: {viz_cls}. Must be a subclass of Viz.")
    # delete and recreate the current viz object with the new type
    delete()
    get()


def get(reconnect: bool = True) -> types.Viz:
    """Get the visualization object.

    Args:
        reconnect (optional): If True, tries to reconnect the visualization object if it is not connected.
            Default: True.

    Returns:
        The visualization object.

    """
    global _viz
    if _viz is None or not isinstance(_viz, _viz_cls):
        _viz = _viz_cls()
    if reconnect:
        _viz.check_status()
    return _viz


def delete() -> None:
    """Remove the visualization object."""
    global _viz
    _viz = None
