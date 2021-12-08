from limbus.core import ComponentsManager
from limbus.components import (  # type: ignore
    limbus___Constant as Constant, limbus___Printer as Printer, limbus___Adder as Adder)

import torch


def test_pipeline():
    c1 = Constant("c1", 2 * torch.ones(1, 3))
    c2 = Constant("c2", torch.ones(1, 3))

    add = Adder("add")
    show = Printer("print")

    manager = ComponentsManager()
    manager.connect(c1, "out", add, "a")
    manager.connect(c2, "out", add, "b")
    manager.connect(add, "out", show, "inp")

    manager.traverse()
    manager.execute(1)

    torch.allclose(add.outputs.out, torch.ones(1, 3) * 3.)
