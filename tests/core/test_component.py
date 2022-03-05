from numpy import isin
import pytest
import typing

import limbus.core.component
from limbus.core import Params, Component, Pipeline, NoValue
from limbus.core.component import Container, Param, IterableContainer, IterableInputContainers, IterableParam
import torch


class TestContainer:
    def test_smoke(self):
        v = Container(None)
        assert isinstance(v, Container)
        assert v.value is None


class TestIterableContainer:
    def test_smoke(self):
        c = IterableContainer(Container(None), 0)
        assert isinstance(c, IterableContainer)
        assert isinstance(c.container, Container)
        assert c.index == 0
        assert c.container.value is None
        # None is not a valid value inside the container. This is not controlled by the container.
        with pytest.raises(TypeError):
            c.value

    def test_value(self):
        c = IterableContainer(Container([1, 2]), 0)
        assert c.value == 1
        assert c.index == 0
        assert c.container.value == [1,2]


class TestIterableInputContainers:
    def test_smoke(self):
        c = IterableInputContainers()
        assert isinstance(c, IterableInputContainers)
        assert c._containers == []

    def test_init_with_container(self):
        ic = IterableContainer(Container(None), 0)
        c = IterableInputContainers(ic)
        assert c._containers[0] is ic
        assert len(c) == 1

    def test_add(self):
        ic0 = IterableContainer(Container(None), 0)
        ic1 = IterableContainer(Container(None), 1)
        c = IterableInputContainers()
        c.add(ic0)
        c.add(ic1)
        assert c._containers[0] is ic0
        assert c._containers[1] is ic1
        assert len(c) == 2

    def test_remove(self):
        ic = IterableContainer(Container(None), 0)
        c = IterableInputContainers()
        c.add(ic)
        assert len(c) == 1
        c.remove(0)
        assert len(c) == 0

    def test_remove_not_found(self):
        ic = IterableContainer(Container(None), 0)
        c = IterableInputContainers()
        c.add(ic)
        assert len(c) == 1
        c.remove(1)
        assert len(c) == 1

    def test_get_ordered(self):
        ic0 = IterableContainer(Container(1), 0)
        ic1 = IterableContainer(Container(2), 1)
        c = IterableInputContainers()
        c.add(ic1)
        c.add(ic0)

        res = c.get_ordered()
        assert len(res) == 2
        assert res[0] == 1
        assert res[1] == 2

    def test_get_ordered_iterable(self):
        ic0 = IterableContainer(IterableContainer(Container([1, 2]), 1), 0)
        ic1 = IterableContainer(Container(3), 1)
        c = IterableInputContainers()
        c.add(ic1)
        c.add(ic0)

        res = c.get_ordered()
        assert len(res) == 2
        assert res[0] == 2
        assert res[1] == 3


class TestParam:
    def test_smoke(self):
        p = Param("a")
        assert isinstance(p, Param)
        assert p.name == "a"
        assert p.type is typing.Any
        assert isinstance(p.value, NoValue)
        assert p.references == set()
        assert p.arg is None
        assert p._is_subscriptable is False

    def test_init_with_type(self):
        p = Param("a", tp=int)
        assert p.type is int
        assert isinstance(p.value, NoValue)

    def test_init_with_value(self):
        p = Param("a", tp=int, value=1)
        assert p.value == 1

    def test_init_with_arg(self):
        p = Param("a", arg="arg0")
        assert p.arg == "arg0"

    def test_set_value(self):
        p = Param("a", tp=int, value=1)
        p.value = 2
        assert p.value == 2
        with pytest.raises(TypeError):
            p.value = "a"

    def test_get_iterable_value(self):
        iter_container = IterableContainer(Container([1, 2]), 0)
        p = Param("a", value=iter_container)
        assert isinstance(p.container, Container)
        assert p.container.value is iter_container
        assert p.value == 1

    def test_get_iterable_input_value(self):
        p = Param("a", tp=typing.List[int])
        p.container = IterableInputContainers(IterableContainer(Container(1), 1))
        p.container.add(IterableContainer(Container(2), 0))
        assert p.value == [2, 1]

    def test_select(self):
        p = Param("a", typing.List[int], value=[1, 2])
        assert p._is_subscriptable
        iter_param = p.select(0)
        assert isinstance(iter_param, IterableParam)
        assert isinstance(iter_param.iter_container, IterableContainer)
        assert iter_param.iter_container.index == 0
        assert iter_param.iter_container.value == 1

    def test_connect_iterparam_param_no_select_raise_error(self):
        p0 = Param("a", typing.List[int])
        p1 = Param("b")
        with pytest.raises(ValueError):
            # mandatory connect with an index
            p0.connect(p1)
        p0.select(0).connect(p1)

    def test_connect_param_iterparam_no_select_raise_error(self):
        p0 = Param("a", tp=typing.List[int])
        p1 = Param("b", value=2)
        with pytest.raises(ValueError):
            # mandatory connect with an index
            p1.connect(p0)
        p1.connect(p0.select(0))

    def test_connect_param_param(self):
        p0 = Param("a", value=1)
        p1 = Param("b")
        p0.connect(p1)
        assert isinstance(p1.container, Container)
        assert p1.value == 1
        assert list(p0._refs.keys()) == [None]
        assert list(p0._refs[None]) == [(p1, None)]
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None]) == [(p0, None)]

    def test_disconnect_param_param(self):
        p0 = Param("a", value=1)
        p1 = Param("b")
        p0.connect(p1)
        p0.disconnect(p1)
        assert isinstance(p1.container, Container)
        assert isinstance(p1.value, NoValue)
        assert list(p0._refs.keys()) == [None]
        assert list(p0._refs[None]) == []
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None]) == []

    def test_connect_disconnect_iterparam_param(self):
        p0 = Param("a", tp=typing.List[int], value=[1, 2])
        p1 = Param("b")
        p0.select(1).connect(p1)
        assert isinstance(p1.container, Container)
        assert p1.value == 2
        assert list(p0._refs.keys()) == [1]
        assert list(p0._refs[1]) == [(p1, None)]
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None]) == [(p0, 1)]

    def test_disconnect_iterparam_param(self):
        p0 = Param("a", tp=typing.List[int], value=[1, 2])
        p1 = Param("b")
        p0.select(1).connect(p1)
        p0.select(1).disconnect(p1)
        assert isinstance(p1.container, Container)
        assert isinstance(p1.value, NoValue)
        assert list(p0._refs.keys()) == [1]
        assert list(p0._refs[1]) == []
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None]) == []

    def test_connect_param_iterparam(self):
        p0 = Param("a", tp=typing.List[int])
        p1 = Param("b", value=1)
        p2 = Param("c", value=2)
        p1.connect(p0.select(1))
        p2.connect(p0.select(0))
        assert isinstance(p0.container, IterableInputContainers)
        assert p0.value == [2, 1]
        assert sorted(list(p0._refs.keys())) == sorted([0, 1])
        assert list(p0._refs[0]) == [(p2, None)]
        assert list(p0._refs[1]) == [(p1, None)]
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None]) == [(p0, 1)]
        assert list(p2._refs.keys()) == [None]
        assert list(p2._refs[None]) == [(p0, 0)]

    def test_disconnect_param_iterparam(self):
        p0 = Param("a", tp=typing.List[int])
        p1 = Param("b", value=1)
        p2 = Param("c", value=2)
        p1.connect(p0.select(1))
        p2.connect(p0.select(0))
        p1.disconnect(p0.select(1))
        p2.disconnect(p0.select(0))
        assert isinstance(p0.container, Container)
        assert isinstance(p0.value, NoValue)
        assert sorted(list(p0._refs.keys())) == sorted([0, 1])
        assert list(p0._refs[0]) == []
        assert list(p0._refs[1]) == []
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None]) == []
        assert list(p2._refs.keys()) == [None]
        assert list(p2._refs[None]) == []

    def test_connect_iterparam_iterparam(self):
        p0 = Param("a", tp=typing.List[int], value=[1, 2])
        p1 = Param("b", tp=typing.List[int])
        p0.select(0).connect(p1.select(1))
        p0.select(1).connect(p1.select(0))
        assert isinstance(p1.container, IterableInputContainers)
        assert p1.value == [2, 1]
        assert sorted(list(p0._refs.keys())) == sorted([0, 1])
        assert list(p0._refs[0]) == [(p1, 1)]
        assert list(p0._refs[1]) == [(p1, 0)]
        assert sorted(list(p1._refs.keys())) == sorted([0, 1])
        assert list(p1._refs[0]) == [(p0, 1)]
        assert list(p1._refs[1]) == [(p0, 0)]

    def test_disconnect_iterparam_iterparam(self):
        p0 = Param("a", tp=typing.List[int], value=[1, 2])
        p1 = Param("b", tp=typing.List[int])
        p0.select(0).connect(p1.select(1))
        p0.select(1).connect(p1.select(0))
        p0.select(0).disconnect(p1.select(1))
        p0.select(1).disconnect(p1.select(0))
        assert isinstance(p1.container, Container)
        assert isinstance(p1.value, NoValue)
        assert sorted(list(p0._refs.keys())) == sorted([0, 1])
        assert list(p0._refs[0]) == []
        assert list(p0._refs[1]) == []
        assert sorted(list(p1._refs.keys())) == sorted([0, 1])
        assert list(p1._refs[0]) == []
        assert list(p1._refs[1]) == []

    def test_ref_count(self):
        p0 = Param("a")
        p1 = Param("b")
        p2 = Param("c")
        p0.connect(p1)
        p0.connect(p2)
        assert p1.ref_counter(None) == 1
        assert p0.ref_counter(None) == 2
        assert p2.ref_counter(None) == 1

    def test_ref_counter(self):
        p0 = Param("a")
        p1 = Param("b")
        p2 = Param("c")
        p0.connect(p1)
        p0.connect(p2)
        assert p1.ref_counter() == 1
        assert p0.ref_counter() == 2
        assert p2.ref_counter() == 1
        assert p1.ref_counter(None) == 1
        assert p0.ref_counter(None) == 2
        assert p2.ref_counter(None) == 1

    def test_ref_counter_iterable(self):
        p0 = Param("a", typing.List[int], value=[1, 2])
        p1 = Param("b")
        p2 = Param("c")
        p3 = Param("d")
        p0.select(0).connect(p1)
        p0.select(1).connect(p2)
        assert p1.ref_counter() == 1
        assert p2.ref_counter() == 1
        assert p0.ref_counter() == 2
        assert p0.ref_counter(0) == 1
        assert p0.ref_counter(1) == 1


class TestIterableParam:
    def test_smoke(self):
        p = Param("a", tp=typing.List[int], value=[1, 2])
        ip = IterableParam(p, 0)
        assert ip.param is p
        assert ip.index == 0
        assert ip.value == 1
        assert isinstance(ip.iter_container, IterableContainer)
        assert ip.iter_container.index == 0
        assert ip.iter_container.value == 1

    def test_ref_counter(self):
        p0 = Param("a", tp=typing.List[int], value=[1, 2])
        ip = IterableParam(p0, 0)
        assert ip.ref_counter() == 0
        p1 = Param("b", tp=int)
        ip.connect(p1)
        assert ip.ref_counter() == 1


class TestParams:
    def test_smoke(self):
        p = Params()
        assert p is not None

    def test_declare(self):
        p = Params()

        with pytest.raises(AttributeError):
            p.x

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

    def test_tensor(self):
        p1 = Params()
        p2 = Params()

        p1.declare("x", torch.Tensor, torch.tensor(1.))
        assert isinstance(p1["x"].value, torch.Tensor)

        p2.declare("y", torch.Tensor, p1.x)
        assert p1.x.value == p2.y.value


class TestComponent:
    def test_smoke(self):
        cmp = Component("yuhu")
        assert cmp.name == "yuhu"
        assert cmp.inputs is not None
        assert cmp.outputs is not None


def test_check_subscriptable():
    assert limbus.core.component._check_subscriptable(typing.Sequence[int])
    assert limbus.core.component._check_subscriptable(typing.Iterable[int])
    assert limbus.core.component._check_subscriptable(typing.List[int])
    assert limbus.core.component._check_subscriptable(typing.Tuple[int])
    assert not limbus.core.component._check_subscriptable(int)
    assert not limbus.core.component._check_subscriptable(torch.Tensor)
    assert not limbus.core.component._check_subscriptable(str)
