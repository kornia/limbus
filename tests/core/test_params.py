import pytest
from typing import Any

import torch

from limbus.core import Params, NoValue, InputParams, OutputParams
from limbus.core.param import Param, InputParam, OutputParam


class TestParams:
    def test_smoke(self):
        p = Params()
        assert p is not None

    def test_declare(self):
        p = Params()

        assert p.x is None  # p.x does not exist but Params accept dynamic attributes

        p.declare("x")
        assert isinstance(p.x.value, NoValue)
        assert isinstance(p.get_param("x"), NoValue)

        p.declare("y", float, 1.)
        assert p.y.value == 1.
        assert p["y"].value == 1.
        assert p.get_param("y") == 1.
        assert isinstance(p["y"], Param)
        assert isinstance(p.y, Param)
        assert isinstance(p["y"].value, float)
        assert p["y"].type == float
        assert p["y"].name == "y"
        assert p["y"].arg is None
        assert p.get_related_arg("y") is None

    def test_tensor(self):
        p1 = Params()
        p2 = Params()

        p1.declare("x", torch.Tensor, torch.tensor(1.))
        assert isinstance(p1["x"].value, torch.Tensor)

        p2.declare("y", torch.Tensor, p1.x)
        assert p1.x.value == p2.y.value

    def test_get_param(self):
        p = Params()
        p.declare("x")
        p.declare("y", float, 1.)
        assert len(p) == 2
        assert p.get_params() == ["x", "y"]
        assert isinstance(p.get_param("x"), NoValue)
        assert p.get_param("y") == 1.
        p.set_param("x", "xyz")
        assert p.get_param("x") == "xyz"

    def test_wrong_set_param_type(self):
        p = Params()
        with pytest.raises(TypeError):
            p.declare("x", int, 1.)
        p.declare("x", int)
        with pytest.raises(TypeError):
            p.set_param("x", "xyz")

    def test_get_type(self):
        p = Params()
        p.declare("x")
        p.declare("y", float, 1.)
        assert p.get_type("x") == Any
        assert p.get_type("y") == float
        assert p.get_types() == {"x": Any, "y": float}


class TestInputParams:
    def test_declare(self):
        p = InputParams()
        p.declare("x", float, 1.)
        assert isinstance(p.x, InputParam)

    def test_declare_with_param(self):
        p = InputParams()
        p0 = Param("x", float, 1.)
        p.declare("x", float, p0)
        assert p.x.value == p0.value


class TestOutputParams:
    def test_declare(self):
        p = OutputParams()
        p.declare("x", float, 1.)
        assert isinstance(p.x, OutputParam)

    def test_declare_with_param(self):
        p = OutputParams()
        p0 = Param("x", float, 1.)
        p.declare("x", float, p0)
        assert p.x.value == p0.value
