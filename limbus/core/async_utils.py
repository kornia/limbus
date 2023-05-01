"""Async utils for limbus."""
from __future__ import annotations
import asyncio
import inspect
from typing import Coroutine, TYPE_CHECKING

if TYPE_CHECKING:
    from limbus.core.component import Component

# Get the loop that is going to run the pipeline. Doing it in this way allows to rerun the pipeline.
loop = asyncio.new_event_loop()


def reset_loop() -> asyncio.AbstractEventLoop:
    """Reset the loop."""
    global loop
    loop = asyncio.new_event_loop()
    return loop


def run_coroutine(coro: Coroutine) -> None:
    """Run a coroutine in an event loop.

    Args:
        coro: coroutine to run.

    """
    global loop
    if loop.is_closed():
        loop = reset_loop()
    loop.run_until_complete(coro)


def get_task_if_exists(component: Component) -> None | asyncio.Task:
    """Get the task associated to a given component if it exists.

    Args:
        component: component to check.

    Returns:
        None | asyncio.Task: task associated to the component if it exists, None otherwise.

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
