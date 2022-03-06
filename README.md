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
c1.outputs.out.connect(add.inputs.a)
c2.outputs.out.connect(add.inputs.b)
add.outputs.out.connect(show.inputs.inp)

# create the pipeline and add its nodes
pipeline = Pipeline()
pipeline.add_nodes([c1, c2, add, show])

# run your pipeline
pipeline.traverse()
pipeline.execute(1)

torch.allclose(add.outputs.out.value, torch.ones(1, 3) * 2.)
```

Example using the `stack` torch method:

```python
# define your components
c1 = Constant("c1", 0)
t1 = Constant("t1", torch.ones(1, 3))
t2 = Constant("t2", torch.ones(1, 3) * 2)
stack = limbus.components.torch.stack("stack")
show = Printer("print")

# connect the components
c1.outputs.out.connect(stack.inputs.dim)
t1.outputs.out.connect(stack.inputs.tensors.select(0))
t2.outputs.out.connect(stack.inputs.tensors.select(1))
stack.outputs.out.connect(show.inputs.inp)

# create the pipeline and add its nodes
pipeline = Pipeline()
pipeline.add_nodes([c1, t1, t2, stack, show])

# run your pipeline
pipeline.traverse()
pipeline.execute(1)

torch.allclose(stack.outputs.out.value, torch.tensor([[1., 1., 1.],[2., 2., 2.]]))
```

Remember that the components can be run without the `Pipeline`, e.g in the last example you can also run:

```python
c1()
t1()
t2()
stack()
show()
```

## Installation

```bash
git clone https://github.com/kornia/limbus
cd limbus
source path.bash.inc
```

In order to regenerate the environment:
```bash
cd limbus
rm -rf .dev_env
```

## Testing

Run `pytest` and automatically will test: `cov`, `pydocstyle`, `mypy` and `flake8`
