from limbus.core import Pipeline
import limbus.components
from limbus.components.base import Constant, Printer, Adder

import torch


# TODO: test in detail the functions
class TestPipeline:
    def test_smoke(self):
        man = Pipeline()
        man is not None


def test_pipeline():
    c1 = Constant("c1", 2 * torch.ones(1, 3))
    c2 = Constant("c2", torch.ones(1, 3))
    add = Adder("add")
    show = Printer("print")

    c1.outputs.out.connect(add.inputs.a)
    c2.outputs.out.connect(add.inputs.b)
    add.outputs.out.connect(show.inputs.inp)

    manager = Pipeline()
    manager.add_nodes([c1, c2, add, show])
    manager.traverse()
    manager.execute(1)

    torch.allclose(add.outputs.out.value, torch.ones(1, 3) * 3.)


def test_pipeline_simple_graph():
    c1 = Constant("c1", torch.rand(2, 3))
    show0 = Printer("print0")
    c1.outputs.out.connect(show0.inputs.inp)
    manager = Pipeline()
    manager.add_nodes([c1, c2])
    manager.traverse()
    manager.execute(1)


def test_pipeline_disconnected_components():
    c1 = Constant("c1", torch.rand(2, 3))
    show0 = Printer("print0")
    c1.outputs.out.connect(show0.inputs.inp)
    c1.outputs.out.disconnect(show0.inputs.inp)
    manager = Pipeline()
    manager.add_nodes([c1, show0])
    manager.traverse()
    manager.execute(1)


def test_pipeline_iterable():
    c1 = Constant("c1", torch.rand(2, 3))
    c2 = Constant("c2", 0)
    unbind = limbus.components.torch.unbind("unbind")
    show0 = Printer("print0")
    c1.outputs.out.connect(unbind.inputs.input)
    c2.outputs.out.connect(unbind.inputs.dim)
    unbind.outputs.out.select(0).connect(show0.inputs.inp)
    manager = Pipeline()
    manager.add_nodes([c1, c2, unbind, show0])
    manager.traverse()
    manager.execute(1)
