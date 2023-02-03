"""Pytest fictures."""
import pytest
import asyncio


@pytest.fixture(scope="class")
def event_loop_instance():
    """Ensure there is an event loop running."""
    try:
        asyncio.get_running_loop()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
