import pytest
import asyncio

from limbus.core.component import ComponentState
from limbus.widgets import viz
from limbus.widgets import is_disabled


@pytest.fixture
def class_with_viz():
    class A:
        @is_disabled
        async def f(self) -> ComponentState:
            return ComponentState.OK
    return A


def _init_viz_module_params(viz_cls):
    viz._viz = None
    viz._viz_cls = viz_cls


VALID_TYPES = [viz.types.Console, viz.types.Visdom]


@pytest.mark.parametrize("value", (VALID_TYPES))
def test_get(value):
    _init_viz_module_params(value)

    # create viz object
    v = viz.get()
    assert v == viz._viz
    assert isinstance(v, viz._viz_cls)

    # validate that viz object is a singleton
    v2 = viz.get()
    assert v == v2


def test_get_with_change_of_cls():
    _init_viz_module_params(viz.types.Console)

    # create viz object
    v = viz.get()
    assert v == viz._viz
    assert isinstance(v, viz._viz_cls)

    viz._viz_cls = viz.types.Visdom

    # validate that viz object is a singleton
    v2 = viz.get()
    assert v != v2
    assert v2 == viz._viz
    assert isinstance(v2, viz._viz_cls)


@pytest.mark.parametrize("value", (VALID_TYPES))
def test_delete(value):
    _init_viz_module_params(value)

    # create init viz module if it is not created
    viz.get()

    # create viz object
    viz.delete()
    assert viz._viz is None


@pytest.mark.parametrize("value", (VALID_TYPES))
def test_set_types_class(value):
    viz.set_type(value)
    assert viz._viz_cls == value
    assert isinstance(viz._viz, value)


@pytest.mark.parametrize("value, viz_type", ([("Console", viz.types.Console), ("Visdom", viz.types.Visdom)]))
def test_set_types_str(value, viz_type):
    viz.set_type(value)
    assert viz._viz_cls == viz_type
    assert isinstance(viz._viz, viz_type)


def test_set_types_new_viz_class():
    class A(viz.types.Viz):
        def check_status(self):
            pass

        def show_image(self):
            pass

        def show_images(self):
            pass

        def show_text(self):
            pass

    viz.set_type(A)
    assert viz._viz_cls == A
    assert isinstance(viz._viz, A)


def test_invalid_viz_type():
    # invalid string
    with pytest.raises(ValueError):
        viz.set_type("InvalidVizType")

    # invalid class type
    class A:
        pass
    with pytest.raises(ValueError):
        viz.set_type(A)

    # base class cannot be passed
    with pytest.raises(ValueError):
        viz.set_type(viz.types.Viz)


# THE NEXT TESTS ARE FOR CODE IN base_component.py

@pytest.mark.parametrize("value", (VALID_TYPES))
def test_decorator_is_disabled_with_viz_disabled(class_with_viz, value):
    _init_viz_module_params(value)
    viz.get()
    # by default, the viz is always enabled in Console class
    viz._viz._enabled = False
    assert asyncio.run(class_with_viz().f()) == ComponentState.DISABLED


@pytest.mark.parametrize("value", (VALID_TYPES))
def test_decorator_is_disabled_with_valid_viz(class_with_viz, value):
    _init_viz_module_params(value)
    if value == viz.types.Visdom:
        # visdom server is not enabled so the state is DISABLED
        assert asyncio.run(class_with_viz().f()) == ComponentState.DISABLED
    else:
        assert asyncio.run(class_with_viz().f()) == ComponentState.OK
