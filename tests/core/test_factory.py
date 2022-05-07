from pathlib import Path

import pytest
import torch

import limbus.components
from limbus.core import Component, ComponentState, Params
from limbus import core


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

    def forward(self) -> ComponentState:  # noqa: D102
        a = self.inputs.get_param("a")
        b = self.inputs.get_param("b")
        self._outputs.set_param("out", a - b)
        return ComponentState.OK


def test_registry():
    # Example of a simple component created from the API
    core.register_component(Subs, "test0.test1")
    subs = limbus.components.test0.test1.Subs("name")  # type: ignore

    subs.inputs.set_param("a", torch.randn(2, 3))
    subs.inputs.set_param("b", torch.randn(2, 3))
    subs()
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
    comp.inputs.set_param("input", torch.randn(2, 3))
    comp.inputs.set_param("dim", 2)
    comp()
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
    comp.inputs.set_param("input", torch.randn(2, 3))
    comp.inputs.set_param("dim", 2)
    comp()
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
    comp.inputs.set_param("input", torch.randn(2, 3))
    comp()
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
    comp.inputs.set_param("input", torch.randn(2, 3))
    comp.inputs.set_param("dim", 2)
    comp()
    assert comp.outputs.get_param("out").shape == torch.Size([2, 3, 1])


def test_registry_from_yml_only_name(tmpdir_factory):
    fn = str(Path(tmpdir_factory.mktemp("test_registry_from_yml")) / "test.yml")
    with open(fn, "w") as f:
        f.write("""
            torch.unsqueeze:
    """)
    core.register_components_from_yml(str(fn))
    comp = limbus.components.torch.unsqueeze("test")
    comp.inputs.set_param("input", torch.randn(2, 3))
    comp.inputs.set_param("dim", 2)
    comp()
    assert comp.outputs.get_param("out").shape == torch.Size([2, 3, 1])


@pytest.fixture(scope="module")
def my_components_yml(tmpdir_factory):
    fn = str(Path(tmpdir_factory.mktemp("my_components_yml")) / "test.yml")
    with open(fn, "w") as f:
        f.write("""
            inspect.isclass:
    """)
    return fn


@pytest.fixture(scope="module")
def my_components_module(tmpdir_factory):
    fn = str(Path(tmpdir_factory.mktemp("my_components_module")) / "test.py")
    with open(fn, "w") as f:
        f.write("""
import torch

from limbus.core import Component, ComponentState, Params

class Subs(Component):
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

    def forward(self) -> ComponentState:  # noqa: D102
        a = self.inputs.get_param("a")
        b = self.inputs.get_param("b")
        self._outputs.set_param("out", a - b)
        return ComponentState.OK
    """)
    return fn


def test_registry_from_module(my_components_module):
    core.register_components_from_module(str(my_components_module))
    comp = limbus.components.test.Subs("test")
    comp.inputs.set_param("a", torch.tensor(3))
    comp.inputs.set_param("b", torch.tensor(2))
    comp()
    assert comp.outputs.get_param("out") == torch.tensor(1)


@pytest.mark.parametrize("path, func", [("my_components_module", "test.Subs"),
                                        ("my_components_yml", "inspect.isclass")])
def test_registry_from_path(path, func, request):
    path = request.getfixturevalue(path)
    core.register_components_from_path(str(path))
    eval(f"limbus.components.{func}")


def test_deregistry_all(my_components_module):
    core.register_components_from_module(str(my_components_module))
    comp = limbus.components.test.Subs("test")
    comp.inputs.set_param("a", torch.tensor(3))
    comp.inputs.set_param("b", torch.tensor(2))
    comp()

    core.deregister_all_components()
    with pytest.raises(AttributeError):
        limbus.components.test.Subs("test")

    # check default modules
    dir(limbus.components.torch)
    dir(limbus.components.kornia)
    dir(limbus.components.base)
