import pytest
from limbus.core import Params, Component, ComponentsManager
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
        assert p.x is None

        p.declare("y", 1.)
        assert p.y == 1.
        assert p["y"] == 1.
    
    def test_tensor(self):
        p1 = Params()
        p2 = Params()

        p1.declare("x", torch.tensor(1.))
        assert isinstance(p1["x"], torch.Tensor)

        p2.declare("y", p1.x)
        assert p1.x == p2.y


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