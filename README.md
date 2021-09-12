# Limbus: Computer Vision pipelining for PyTorch

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

# create the manager and connect them
manager = ComponentsManager()
manager.connect(c1, "out", add, "a")
manager.connect(c2, "out", add, "b")
manager.connect(add, "out", show, "inp")

# run your pipeline
manager.traverse()
manager.execute(1)

torch.allclose(add.outputsout, torch.ones(1, 3) * 2.)
```

## Installation

```bash
git clone https://github.com/kornia/limbus
cd limbus
source path.bash.inc
```
