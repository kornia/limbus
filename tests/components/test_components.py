import pytest

from limbus.components.base import Constant, Adder  # type: ignore
from limbus.core import NoValue
import torch


class TestConstant:
    @pytest.mark.parametrize("value", ([1, 2., torch.tensor(3.)]))
    def test_smoke(self, value):
        c = Constant("k", value)
        assert c.name == "k"
        assert isinstance(c.outputs.out, NoValue)

        c.forward(c.inputs)
        assert c.outputs.out == value


class TestAdder:
    def test_smoke(self):
        add = Adder("add")
        assert add.name == "add"
        assert isinstance(add.outputs.out, NoValue)

        add.inputs.a = torch.tensor(2.)
        add.inputs.b = torch.tensor(3.)
        add.forward(add.inputs)
        assert add.outputs.out == torch.tensor(5.)
