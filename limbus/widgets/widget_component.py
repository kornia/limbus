"""Module containing the base component for visualization components."""
import functools
from abc import abstractmethod
from typing import Callable
from enum import Enum

from limbus import widgets
from limbus.core import Component, ComponentState, PropParams


class WidgetState(Enum):
    """Possible states for the viz."""
    DISABLED = 0  # viz is disabled but can be enabled.
    ENABLED = 1  # viz is enabled.
    NO = 2  # viz cannot be used.


# this is a decorator that will return ComponentState.DISABLED if the visualization is not enabled.
def is_disabled(func: Callable) -> Callable:
    """Return ComponentState.DISABLED if viz is not enabled."""
    @functools.wraps(func)
    async def wrapper_check_component_disabled(self, *args, **kwargs):
        vz = widgets.get(False)
        if vz is None or not vz.enabled:
            return ComponentState.DISABLED
        return await func(self, *args, **kwargs)
    return wrapper_check_component_disabled


class WidgetComponent(Component):
    """Allow to use widgets in Limbus Components.

    Args:
        name (str): component name.

    """
    # By default the components do not have viz.
    # Default WIDGET_STATE must be static because we need to get access when the class is not instantiated.
    # To change the widget state, use the widget_state property.
    WIDGET_STATE: WidgetState = WidgetState.ENABLED

    def __init__(self, name: str):
        super().__init__(name)
        self._widget_state: WidgetState = self.__class__.WIDGET_STATE

    @property
    def widget_state(self) -> WidgetState:
        """Get the viz state for this component."""
        return self._widget_state

    @widget_state.setter
    def widget_state(self, state: WidgetState) -> None:
        """Set the viz state for this component."""
        self._widget_state = state


class BaseWidgetComponent(WidgetComponent):
    """Base class for only visualization components.

    Args:
        name (str): component name.

    """
    # by default Widget Components have the viz enabled, to disable it use the widget_state property.
    WIDGET_STATE: WidgetState = WidgetState.ENABLED

    @staticmethod
    def register_properties(properties: PropParams) -> None:
        """Register the properties.

        Args:
             properties: object to register the properties.

        """
        # this line is like super() but for static methods.
        Component.register_properties(properties)
        properties.declare("title", str, "")

    @abstractmethod
    async def _show(self, title: str) -> None:
        """Show the data.

        Args:
            title: same as self._properties.get_param("title").

        """
        raise NotImplementedError

    @is_disabled
    async def forward(self) -> ComponentState:  # noqa: D102
        await self._show(self._properties.get_param("title"))
        return ComponentState.OK
