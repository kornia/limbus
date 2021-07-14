from limbus.core import ComponentsManager
from limbus.components import Constant, Adder, Printer

import torch


def test_pipeline():
    c1 = Constant("c1", 1.)
    c2 = Constant("c2", torch.ones(1, 3))

    add = Adder("add")
    show = Printer("print")

    manager = ComponentsManager()
    manager.connect(c1, "out", add, "a")
    manager.connect(c2, "out", add, "b")
    manager.connect(add, "sum_out", show, "inp")

    # TODO: traverse should be called automatically with execute
    manager.traverse()
    manager.execute()

    torch.allclose(add.outputs.sum_out, torch.ones(1, 3) * 2.)

    # TODO: reset nodes
    # manager.connect(c2, "out", add, "a")
    # manager.traverse()
    # manager.execute()