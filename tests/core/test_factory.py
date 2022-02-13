import torch

import limbus.components
from limbus.core import Component, ComponentState, Params, register_component


def test_registry():
    # Example of a simple component created from the API
    class Subs(Component):
        """Component to add two inputs and output the result."""
        def __init__(self, name: str):
            super().__init__(name)

        @staticmethod
        def register_inputs() -> Params:  # noqa: D102
            inputs = Params()
            inputs.declare("a", torch.Tensor)
            inputs.declare("b", torch.Tensor)
            return inputs

        @staticmethod
        def register_outputs() -> Params:  # noqa: D102
            outputs = Params()
            outputs.declare("out", torch.Tensor)
            return outputs

        def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
            a = inputs.get_param("a")
            b = inputs.get_param("b")
            self._outputs.set_param("out", a - b)
            return ComponentState.OK

    register_component(Subs, "test0.test1")
    subs = limbus.components.test0.test1.Subs("name")  # type: ignore

    inp = Params()
    inp.declare("a", torch.Tensor, torch.randn(2, 3))
    inp.declare("b", torch.Tensor, torch.randn(2, 3))
    subs(inp)
    assert subs.outputs.get_param("out").shape == torch.Size([2, 3])
