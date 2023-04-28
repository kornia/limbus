"""Pytest fictures."""
import pytest
import asyncio

from limbus.core import async_utils


@pytest.fixture
def event_loop_instance():
    """Ensure there is an event loop running."""
    if async_utils.loop.is_closed():
        async_utils.reset_loop()
    asyncio.set_event_loop(async_utils.loop)
