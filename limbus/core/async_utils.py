"""Async utils for limbus."""
from __future__ import annotations
import asyncio
import inspect
from typing import Coroutine, TYPE_CHECKING, Optional
from sys import version_info

if TYPE_CHECKING:
    from limbus.core.component import Component


def run_coroutine(coro: Coroutine) -> None:
    """Run a coroutine in an event loop.

    Args:
        coro: coroutine to run.

    """
    if version_info.major != 3:
        raise ValueError("Only python 3 is supported.")
    if version_info.minor < 10:
        # for python <3.10 the loop must be run in this way to avoid creating a new loop.
        loop = asyncio.get_event_loop()
        loop.run_until_complete(coro)
    elif version_info.minor >= 10:
        # for python >=3.10 the loop should be run in this way.
        asyncio.run(coro)


def get_task_if_exists(component: Component) -> Optional[asyncio.Task]:
    """Get the task associated to a given component if it exists.

    Args:
        component: component to check.

    Returns:
        Optional[asyncio.Task]: task associated to the component if it exists, None otherwise.

    """
    task: asyncio.Task
    for task in asyncio.all_tasks():
        coro = task.get_coro()
        assert isinstance(coro, Coroutine)  # added to avoid mypy issues
        cr_locals = inspect.getcoroutinelocals(coro)
        # check if the coroutine of the component object already exists in the tasks list
        if "self" in cr_locals and cr_locals["self"] is component:
            return task
    return None


def check_if_task_exists(component: Component) -> bool:
    """Check if the coroutine of the parent object already exists in the tasks list.

    Args:
        component: parent component object to check.

    Returns:
        True if the coroutine of the component object already exists in the tasks list, False otherwise.

    """
    if get_task_if_exists(component) is not None:
        return True
    return False


def create_task_if_needed(ref_component: Component, component: Component) -> None:
    """Create the task for the component if it is not created yet.

    Args:
        ref_component: reference component.
        component: component to create the task.

    """
    if not check_if_task_exists(component):
        # start the execution of the component if it is not started yet
        component.init_from_component(ref_component)
        asyncio.create_task(component())
