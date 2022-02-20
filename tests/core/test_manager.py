from limbus.core import ComponentsManager
from limbus.components.base import Constant, Printer, Adder

import torch


def test_pipeline():
    c1 = Constant("c1", 2 * torch.ones(1, 3))
    c2 = Constant("c2", torch.ones(1, 3))
    add = Adder("add")
    show = Printer("print")

    c1.outputs.out.connect(add.inputs.a)
    c2.outputs.out.connect(add.inputs.b)
    add.outputs.out.connect(show.inputs.inp)

    manager = ComponentsManager()
    manager.add([c1, c2, add, show])
    manager.traverse()
    manager.execute(1)

    torch.allclose(add.outputs.out.value, torch.ones(1, 3) * 3.)
