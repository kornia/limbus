import pytest

from limbus.components import Constant, Adder
import torch


class TestConstant:
    @pytest.mark.parametrize("value", ([1, 2., torch.tensor(3.)]))
    def test_smoke(self, value):
        c = Constant("k", value)
        assert c.name == "k"
        assert c.outputs.out is None

        c.forward(c.inputs, c.outputs)
        assert c.outputs.out == value


class TestAdder:
    def test_smoke(sel):
        add = Adder("add")
        assert add.name == "add"
        assert add.outputs.sum_out is None

        add.inputs.a = 2.
        add.inputs.b = torch.tensor(3.)
        add.forward(add.inputs, add.outputs)
        assert add.outputs.sum_out == torch.tensor(5.)