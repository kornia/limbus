"""Basic example defining components and connecting them."""
from typing import List, Any
import asyncio

# If you want to change the limbus config you need to do it before importing any limbus module!!!
from limbus_config import config
config.COMPONENT_TYPE = "torch"

from limbus.core import Component, Params, InputParams, OutputParams, ComponentState, VerboseMode  # noqa: E402
from limbus.core.pipeline import Pipeline  # noqa: E402


# define the components
# ---------------------
class Add(Component):
    """Add two numbers."""
    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("a", int)
        inputs.declare("b", int)

    @staticmethod
    def register_outputs(outputs: Params) -> None:  # noqa: D102
        outputs.declare("out", int)

    async def forward(self) -> ComponentState:  # noqa: D102
        a, b = await asyncio.gather(self._inputs.a.receive(), self._inputs.b.receive())
        print(f"Add: {a} + {b}")
        await self._outputs.out.send(a + b)
        return ComponentState.OK


class Printer(Component):
    """Prints the input to the console."""
    @staticmethod
    def register_inputs(inputs: Params) -> None:  # noqa: D102
        inputs.declare("inp", Any)

    async def forward(self) -> ComponentState:  # noqa: D102
        value = await self._inputs.inp.receive()
        print(f"Printer: {value}")
        return ComponentState.OK


class Data(Component):
    """Data source of inf numbers."""
    def __init__(self, name: str, initial_value: int = 0):
        super().__init__(name)
        self._initial_value: int = initial_value

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
        outputs.declare("out", int)

    async def forward(self) -> ComponentState:  # noqa: D102
        print(f"Read: {self._initial_value}")
        await self._outputs.out.send(self._initial_value)
        self._initial_value += 1
        return ComponentState.OK


class Acc(Component):
    """Accumulate data in a list."""
    def __init__(self, name: str, elements: int = 1):
        super().__init__(name)
        self._elements: int = elements

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("inp", int)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
        outputs.declare("out", List[int])

    async def forward(self) -> ComponentState:  # noqa: D102
        res: List[int] = []
        while len(res) < self._elements:
            res.append(await self._inputs.inp.receive())
            print(f"Acc {len(res)}: {res}")

        print(f"Acc: {res}")
        await self._outputs.out.send(res)
        return ComponentState.OK


# create the components
# ---------------------
data0 = Data("data0", 0)
data10 = Data("data10", 10)
add = Add("add")
acc = Acc(name="acc", elements=2)
printer0 = Printer("printer0")
printer1 = Printer("printer1")
printer2 = Printer("printer2")

# connect the components
# ----------------------
data0.outputs.out >> add.inputs.a
data10.outputs.out >> add.inputs.b
add.outputs.out >> acc.inputs.inp
acc.outputs.out >> printer2.inputs.inp  # print the accumulated values once all are received
data0.outputs.out >> printer0.inputs.inp  # print the first value (data0)
add.outputs.out >> printer1.inputs.inp  # print the sum of the values (data10 + data0)

# create and run the pipeline
# ---------------------------
engine: Pipeline = Pipeline()
# at least we need to add one node, the others are added automatically
engine.add_nodes([add, printer0])
# there are several states for each component, with this verbose mode we can see them
engine.set_verbose_mode(VerboseMode.COMPONENT)
# run all teh components at least once (since there is an accumulator, some components will be run more than once)
engine.run(1)
