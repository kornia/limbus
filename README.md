# Limbus: Computer Vision pipelining for PyTorch

## (🚨 Warning: Unstable Prototype 🚨)

[![CI](https://github.com/kornia/limbus/actions/workflows/ci.yml/badge.svg)](https://github.com/kornia/limbus/actions/workflows/ci.yml)

Similar to the eye [*corneal limbus*](https://en.wikipedia.org/wiki/Corneal_limbus) - **Limbus** is a framework to create Computer Vision pipelines within the context of Deep Learning and writen in terms of differentiable tensors message passing on top of Kornia and PyTorch.

## Overview

You can create pipelines using the as a base the `limbus.Component` as follows:

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

torch.allclose(add.outputsout, torch.ones(1, 3) * 2.)
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
