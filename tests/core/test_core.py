import pytest

from limbus.core import Params, Component, ComponentsManager, NoValue
from limbus.core.component import Param
import torch


class TestParams:
    def test_smoke(self):
        p = Params()
        assert p is not None

    def test_declare(self):
        p = Params()

        with pytest.raises(AttributeError):
            p.x

        p.declare("x")
        assert isinstance(p.x.value, NoValue)
        assert isinstance(p.get_param("x"), NoValue)

        p.declare("y", float, 1.)
        assert p.y.value == 1.
        assert p["y"].value == 1.
        assert p.get_param("y") == 1.
        assert isinstance(p["y"], Param)
        assert isinstance(p.y, Param)
        assert isinstance(p["y"].value, float)
        assert p["y"].type == float
        assert p["y"].name == "y"
        assert p["y"].arg is None

    def test_tensor(self):
        p1 = Params()
        p2 = Params()

        p1.declare("x", torch.Tensor, torch.tensor(1.))
        assert isinstance(p1["x"].value, torch.Tensor)

        p2.declare("y", torch.Tensor, p1.x)
        assert p1.x.value == p2.y.value


class TestComponent:
    def test_smoke(self):
        cmp = Component("yuhu")
        assert cmp.name == "yuhu"
        assert cmp.inputs is not None
        assert cmp.outputs is not None


# TODO: test in detail the functions
class TestComponentsManager:
    def test_smoke(self):
        man = ComponentsManager()
        man is not None
