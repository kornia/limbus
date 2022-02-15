from pathlib import Path

import torch

import limbus.components
from limbus.core import Component, ComponentState, Params
from limbus import core


def test_registry():
    # Example of a simple component created from the API
    class Subs(Component):
        """Component to add two inputs and output the result."""
        def __init__(self, name: str):
            super().__init__(name)

        @staticmethod
        def register_inputs() -> Params:  # noqa: D102
            inputs = Params()
            inputs.declare("a", torch.Tensor)
            inputs.declare("b", torch.Tensor)
            return inputs

        @staticmethod
        def register_outputs() -> Params:  # noqa: D102
            outputs = Params()
            outputs.declare("out", torch.Tensor)
            return outputs

        def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
            a = inputs.get_param("a")
            b = inputs.get_param("b")
            self._outputs.set_param("out", a - b)
            return ComponentState.OK

    core.register_component(Subs, "test0.test1")
    subs = limbus.components.test0.test1.Subs("name")  # type: ignore

    inp = Params()
    inp.declare("a", torch.Tensor, torch.randn(2, 3))
    inp.declare("b", torch.Tensor, torch.randn(2, 3))
    subs(inp)
    assert subs.outputs.get_param("out").shape == torch.Size([2, 3])


def test_registry_from_yml_with_args(tmpdir_factory):
    fn = str(Path(tmpdir_factory.mktemp("test_registry_from_yml")) / "test.yml")
    with open(fn, "w") as f:
        f.write("""
            torch.unsqueeze:
                params: {input: torch.Tensor, dim: int}
                returns: {out: torch.Tensor}
    """)
    core.register_components_from_yml(str(fn))
    comp = limbus.components.torch.unsqueeze("test")
    inp = Params()
    inp.declare("input", torch.Tensor, torch.randn(2, 3))
    inp.declare("dim", int, 2)
    comp(inp)
    assert comp.outputs.get_param("out").shape == torch.Size([2, 3, 1])


def test_registry_from_yml_with_args_changing_output_name(tmpdir_factory):
    fn = str(Path(tmpdir_factory.mktemp("test_registry_from_yml")) / "test.yml")
    with open(fn, "w") as f:
        f.write("""
            torch.unsqueeze:
                params: {input: torch.Tensor, dim: int}
                returns: {my_name: torch.Tensor}
    """)
    core.register_components_from_yml(str(fn))
    comp = limbus.components.torch.unsqueeze("test")
    inp = Params()
    inp.declare("input", torch.Tensor, torch.randn(2, 3))
    inp.declare("dim", int, 2)
    comp(inp)
    assert comp.outputs.get_param("my_name").shape == torch.Size([2, 3, 1])


def test_registry_from_yml_with_args_default_value(tmpdir_factory):
    fn = str(Path(tmpdir_factory.mktemp("test_registry_from_yml")) / "test.yml")
    with open(fn, "w") as f:
        f.write("""
            torch.unsqueeze:
                params: {input: torch.Tensor, dim: int = 2}
                returns: {out: torch.Tensor}
    """)
    core.register_components_from_yml(str(fn))
    comp = limbus.components.torch.unsqueeze("test")
    inp = Params()
    inp.declare("input", torch.Tensor, torch.randn(2, 3))
    inp.declare("dim", int, comp.inputs.dim)  # TODO: default params shouldn't be explicitly declared
    comp(inp)
    assert comp.outputs.get_param("out").shape == torch.Size([2, 3, 1])


def test_registry_from_yml_with_signature_idx(tmpdir_factory):
    fn = str(Path(tmpdir_factory.mktemp("test_registry_from_yml")) / "test.yml")
    with open(fn, "w") as f:
        f.write("""
            torch.unsqueeze:
                idx: 0
    """)
    core.register_components_from_yml(str(fn))
    comp = limbus.components.torch.unsqueeze("test")
    inp = Params()
    inp.declare("input", torch.Tensor, torch.randn(2, 3))
    inp.declare("dim", int, 2)
    comp(inp)
    assert comp.outputs.get_param("out").shape == torch.Size([2, 3, 1])


def test_registry_from_yml_only_name(tmpdir_factory):
    fn = str(Path(tmpdir_factory.mktemp("test_registry_from_yml")) / "test.yml")
    with open(fn, "w") as f:
        f.write("""
            torch.unsqueeze:
    """)
    core.register_components_from_yml(str(fn))
    comp = limbus.components.torch.unsqueeze("test")
    inp = Params()
    inp.declare("input", torch.Tensor, torch.randn(2, 3))
    inp.declare("dim", int, 2)
    comp(inp)
    assert comp.outputs.get_param("out").shape == torch.Size([2, 3, 1])
