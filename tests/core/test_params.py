import pytest

import torch

from limbus.core import PropertyParams, NoValue, InputParams, OutputParams
from limbus.core.param import Param, InputParam, OutputParam, PropertyParam


class TParams(InputParams):
    """Test class to test Params class with all teh posible params for Param.

    NOTE: Inherits from InputParams because it is the only one that allows to use all the args in Param.

    """
    pass


class TestParams:
    def test_smoke(self):
        p = TParams()
        assert p is not None

    def test_declare(self):
        p = TParams()
        p.declare("x")
        assert isinstance(p.x.value, NoValue)
        assert isinstance(p["x"].value, NoValue)

        p.declare("y", float, 1.)
        assert p.y.value == 1.
        assert p["y"].value == 1.
        assert isinstance(p["y"], Param)
        assert isinstance(p.y, Param)
        assert isinstance(p["y"].value, float)
        assert p["y"].type == float
        assert p["y"].name == "y"
        assert p["y"].arg is None
        assert p.y.arg is None

    def test_tensor(self):
        p1 = TParams()
        p2 = TParams()

        p1.declare("x", torch.Tensor, torch.tensor(1.))
        assert isinstance(p1["x"].value, torch.Tensor)

        p2.declare("y", torch.Tensor, p1.x)
        assert p1.x.value == p2.y.value

    def test_get_params(self):
        p = TParams()
        p.declare("x")
        p.declare("y", float, 1.)
        assert len(p) == 2
        assert p.get_params() == ["x", "y"]
        assert isinstance(p.x.value, NoValue)
        assert p.y.value == 1.
        p.x.value = "xyz"
        assert p.x.value == "xyz"

    def test_wrong_set_param_type(self):
        p = TParams()
        with pytest.raises(TypeError):
            p.declare("x", int, 1.)
        p.declare("x", int)
        with pytest.raises(TypeError):
            p.x.value = "xyz"


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
        assert p.z is None  # Intellisense asumes p.z exist as an InputParams


class TestOutputParams:
    def test_declare(self):
        p = OutputParams()
        p.declare("x", float)
        assert isinstance(p.x, OutputParam)
        assert p.z is None  # Intellisense asumes p.z exist as an OutputParam


class TestPropertyParams:
    def test_declare(self):
        p = PropertyParams()
        p.declare("x", float, 1.)
        assert isinstance(p.x, PropertyParam)
        assert p.z is None  # Intellisense asumes p.z exist as an PropParams

    def test_declare_with_param(self):
        p = PropertyParams()
        p0 = Param("x", float, 1.)
        p.declare("x", float, p0)
        assert p.x.value == p0.value
