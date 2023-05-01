"""Basic example defining components and connecting them."""
from typing import List, Any
import asyncio

# If you want to change the limbus config you need to do it before importing any limbus module!!!
from limbus_config import config
config.COMPONENT_TYPE = "torch"

from limbus.core import (Component, InputParams, OutputParams, PropertyParams, Pipeline, VerboseMode,  # noqa: E402
                         ComponentState, OutputParam, InputParam, async_utils)  # noqa: E402


# define the components
# ---------------------
class Add(Component):
    """Add two numbers."""
    # NOTE: type definition is optional, but it helps with the intellisense. ;)
    class InputsTyping(OutputParams):  # noqa: D106
        a: InputParam
        b: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    async def val_rec_a(self, value: Any) -> Any:  # noqa: D102
        print(f"CALLBACK: Add.a: {value}.")
        return value

    async def val_rec_b(self, value: Any) -> Any:  # noqa: D102
        print(f"CALLBACK: Add.b: {value}.")
        return value

    async def val_sent(self, value: Any) -> Any:  # noqa: D102
        print(f"CALLBACK: Add.out: {value}.")
        return value

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("a", int, callback=Add.val_rec_a)
        inputs.declare("b", int, callback=Add.val_rec_b)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
        outputs.declare("out", int, callback=Add.val_sent)

    async def forward(self) -> ComponentState:  # noqa: D102
        a, b = await asyncio.gather(self._inputs.a.receive(), self._inputs.b.receive())
        print(f"Add: {a} + {b}")
        await self._outputs.out.send(a + b)
        return ComponentState.OK


class Printer(Component):
    """Prints the input to the console."""
    # NOTE: type definition is optional, but it helps with the intellisense. ;)
    class InputsTyping(OutputParams):  # noqa: D106
        inp: InputParam

    inputs: InputsTyping  # type: ignore

    async def val_changed(self, value: Any) -> Any:  # noqa: D102
        print(f"CALLBACK: Printer.inp: {value}.")
        return value

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:  # noqa: D102
        inputs.declare("inp", Any, callback=Printer.val_changed)

    async def forward(self) -> ComponentState:  # noqa: D102
        value = await self._inputs.inp.receive()
        print(f"Printer: {value}")
        return ComponentState.OK


class Data(Component):
    """Data source of inf numbers."""
    # NOTE: type definition is optional, but it helps with the intellisense. ;)
    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    outputs: OutputsTyping  # type: ignore

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
    # NOTE: type definition is optional, but it helps with the intellisense. ;)
    class InputsTyping(OutputParams):  # noqa: D106
        inp: InputParam

    class OutputsTyping(OutputParams):  # noqa: D106
        out: OutputParam

    inputs: InputsTyping  # type: ignore
    outputs: OutputsTyping  # type: ignore

    def __init__(self, name: str, elements: int = 1):
        super().__init__(name)
        self._elements: int = elements

    async def set_elements(self, value: int) -> int:  # noqa: D102
        print(f"CALLBACK: Acc.elements: {value}.")
        # this is a bir tricky since the value is stored in 2 places the property and the variable.
        # Since the acc uses the _elements variable in the forward method we need to update it here
        # as well. Thanks to the callback we do not need to worry about both sources.
        self._elements = value
        return value

    @staticmethod
    def register_properties(properties: PropertyParams) -> None:  # noqa: D102
        properties.declare("elements", int, callback=Acc.set_elements)

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
# run all the components at least 2 times (since there is an accumulator, some components will be run more than once)


async def run() -> None:  # noqa: D103
    await engine.async_run(1)
    await acc.properties.elements.set_property(3)  # change the number of elements to accumulate
    await engine.async_run(1)

async_utils.run_coroutine(run())
