# Limbus: Computer Vision pipelining for PyTorch

[![CI](https://github.com/kornia/limbus/actions/workflows/ci.yml/badge.svg)](https://github.com/kornia/limbus/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/limbus.svg)](https://pypi.org/project/limbus)

Similar to the eye [*corneal limbus*](https://en.wikipedia.org/wiki/Corneal_limbus) - **Limbus** is a framework to create Computer Vision pipelines within the context of Deep Learning and writen in terms of differentiable tensors message passing on top of Kornia and PyTorch.

## Overview

You can create pipelines using `limbus.Component`s as follows:

```python
# define your components
c1 = Constant("c1", 1.)
c2 = Constant("c2", torch.ones(1, 3))
add = Adder("add")
show = Printer("print")

# connect the components
c1.outputs.out >> add.inputs.a
c2.outputs.out >> add.inputs.b
add.outputs.out >> show.inputs.inp

# create the pipeline and add its nodes
pipeline = Pipeline()
pipeline.add_nodes([c1, c2, add, show])

# run your pipeline
pipeline.run(1)

torch.allclose(add.outputs.out.value, torch.ones(1, 3) * 2.)
```

Example using the `stack` torch method:

```python
# define your components
c1 = Constant("c1", 0)
t1 = Constant("t1", torch.ones(1, 3))
t2 = Constant("t2", torch.ones(1, 3) * 2)
stack = Stack("stack")
show = Printer("print")

# connect the components
c1.outputs.out >> stack.inputs.dim
t1.outputs.out >> stack.inputs.tensors.select(0)
t2.outputs.out >> stack.inputs.tensors.select(1)
stack.outputs.out >> show.inputs.inp

# create the pipeline and add its nodes
pipeline = Pipeline()
pipeline.add_nodes([c1, t1, t2, stack, show])

# run your pipeline
pipeline.run(1)

torch.allclose(stack.outputs.out.value, torch.tensor([[1., 1., 1.],[2., 2., 2.]]))
```

Remember that the components can be run without the `Pipeline`, e.g in the last example you can also run:

```python
asyncio.run(asyncio.gather(c1(), t1(), t2(), stack(), show()))
```

Basically, `Pipeline` objects allow you to control the execution flow, e.g. you can stop, pause, resume the execution, determine the number of executions to be run...

A higher level API on top of `Pipeline` is `App` allowing to encapsulate some code. E.g.:

```python
class MyApp(App):
    def create_components(self):
        self.c1 = Constant("c1", 0)
        self.t1 = Constant("t1", torch.ones(1, 3))
        self.t2 = Constant("t2", torch.ones(1, 3) * 2)
        self.stack = stack("stack")
        self.show = Printer("print")

    def connect_components(self):
        self.c1.outputs.out >> self.stack.inputs.dim
        self.t1.outputs.out >> self.stack.inputs.tensors.select(0)
        self.t2.outputs.out >> self.stack.inputs.tensors.select(1)
        self.stack.outputs.out >> self.show.inputs.inp

MyApp().run(1)
```

## Component definition

Creating your own components is pretty easy, you just need to inherit from `limbus.Component` and implement some methods (see some examples in `examples/defining_cmps.py`).

The `Component` class has the next main methods:
- `__init__`: where you can add class parameters to your component.
- `register_inputs`: where you need to declare the input pins of your component.
- `register_outputs`: where you need to declare the output pins of your component.
- `register_properties`: where you can declare properties that can be changed during the execution.
- `forward`: where you must define the logic of your component (mandatory).

For a detailed list of `Component` methods and attributes, please check `limbus/core/component.py`.

**Note** that if you want intellisense (at least in `VSCode` you will need to define the `input` and `output` types).

Let's see a very simple example that sums 2 integers:

```python
class Add(Component):
    """Add two numbers."""
    # NOTE: type definition is optional, but it helps with the intellisense. ;)
    class InputsTyping(OutputParams):
        a: InputParam
        b: InputParam

    class OutputsTyping(OutputParams):
        out: OutputParam

    inputs: InputsTyping
    outputs: OutputsTyping

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        # Here you need to declare the input parameters and their default values (if they have).
        inputs.declare("a", int)
        inputs.declare("b", int)

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        # Here you need to declare the output parameters.
        outputs.declare("out", int)

    async def forward(self) -> ComponentState:
        # Here you must to define the logic of your component.
        a, b = await asyncio.gather(
            self.inputs.a.receive(),
            self.inputs.b.receive()
        )
        await self.outputs.out.send(a + b)
        return ComponentState.OK
```

**Note** that `Component` can inherint from `nn.Module`. By default inherints from `object`.

To change the inheritance, before importing any other `limbus` module, set the `COMPONENT_TYPE` variable as:

```python
from limbus_config import config
config.COMPONENT_TYPE = "torch"
```

## Ecosystem

Limbus is a core technology to easily build different components and create generic pipelines. In the following list, you can find different examples 
about how to use Limbus with some first/third party projects containing components:

- Official examples:
  - Basic pipeline generation: https://github.com/kornia/limbus/blob/main/examples/default_cmps.py
  - Define custom components: https://github.com/kornia/limbus/blob/main/examples/defining_cmps.py
  - Create a web camera application: https://github.com/kornia/limbus/blob/main/examples/defining_cmps.py
- Official repository with a set of basic components: https://github.com/kornia/limbus-components
- Example combining limbus and the farm-ng Amiga: https://github.com/edgarriba/amiga-limbus-examples
- Example implementing a Kornia face detection pipeline: https://github.com/edgarriba/limbus-face-detector

## Installation

### from PyPI:
```bash
pip install limbus  # limbus alone
# or
pip install limbus[components]  # limbus + some predefined components
```

Note that to use widgets you need to install their dependencies:
```bash
pip install limbus[widgets]
```

### from the repository:

```bash
pip install limbus@git+https://git@github.com/kornia/limbus.git  # limbus alone
# or
pip install limbus[components]@git+https://git@github.com/kornia/limbus.git  # limbus + some predefined components
```

### for development

you can install the environment with the following commands:

```bash
git clone https://github.com/kornia/limbus
cd limbus
source path.bash.inc
```

In order to regenerate the development environment:
```bash
cd limbus
rm -rf .dev_env
source path.bash.inc
```

## Testing

Run `pytest` and automatically will test: `cov`, `pydocstyle`, `mypy` and `flake8`
