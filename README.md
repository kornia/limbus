# Limbus: Computer Vision pipelining for PyTorch

## (🚨 Warning: Unstable Prototype 🚨)

[![CI](https://github.com/kornia/limbus/actions/workflows/ci.yml/badge.svg)](https://github.com/kornia/limbus/actions/workflows/ci.yml)

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

## Installation

### from PyPI:
```bash
pip install limbus  # limbus alone
```

optionally some predefined components can be installed.

In the near future with PyPI (not yet available)
```bash
pip install limbus[components]  # limbus + some predefined components
```

currently they can be installed from the repository:
```bash
pip install limbus-components@git+https://git@github.com/kornia/limbus-components.git
```

### from the repository:

```bash
pip install limbus@git+https://git@github.com/kornia/limbus.git  # limbus alone
pip install limbus-components@git+https://git@github.com/kornia/limbus-components.git  # some predefined components (optional)
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
