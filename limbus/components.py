from typing import Any

from limbus.core import Component, ComponentState, Params


class Constant(Component):
    """Component that holds a constant."""
    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self._value = value
        self._outputs.declare("out")

    def forward(self, inputs: Params, outputs: Params) -> ComponentState:
        outputs.out = self._value
        return ComponentState.OK


class Printer(Component):
    """Component to print the input in the console."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("inp")

    def forward(self, inputs: Params, outputs: Params) -> ComponentState:
        print(inputs.inp)
        return ComponentState.OK


class Adder(Component):
    """Component to add two input and output the result."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("a")
        self._inputs.declare("b")
        self._outputs.declare("sum_out")

    def forward(self, inputs: Params, outputs: Params) -> ComponentState:
        outputs.sum_out = inputs.a + inputs.b
        return ComponentState.OK