import pytest
from typing import Any, List, Sequence, Iterable, Tuple
import asyncio

import torch

import limbus.core.param
from limbus.core import NoValue, Component, ComponentState
from limbus.core.param import (Container, Param, InputParam, OutputParam,
                               IterableContainer, IterableInputContainers, IterableParam, Reference)


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
        assert c.container.value == [1, 2]

    def test_value_with_iterable(self):
        c = IterableContainer(IterableContainer(Container([1, 2]), 1), 0)
        assert c.value == 2
        assert c.index == 0
        assert c.container.value == 2


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
        ic1 = IterableContainer(Container(2), 2)
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


@pytest.mark.usefixtures("event_loop_instance")
class TestParam:
    def test_smoke(self):
        p = Param("a")
        assert isinstance(p, Param)
        assert p.name == "a"
        assert p.parent is None
        assert p.type is Any
        assert isinstance(p.value, NoValue)
        assert p.references == set()
        assert p.arg is None
        assert p._is_subscriptable is False
        assert p.is_subscriptable is False
        assert p() == p.value

    def test_subcriptability(self):
        p = Param("a", List[torch.Tensor], value=[torch.tensor(1), torch.tensor(1)])
        assert p._is_subscriptable
        assert p.is_subscriptable
        p.set_as_non_subscriptable()
        assert not p._is_subscriptable
        p.reset_is_subscriptable()
        assert p._is_subscriptable

    def test_init_with_type(self):
        p = Param("a", tp=int)
        assert p.type is int
        assert isinstance(p.value, NoValue)

    def test_init_with_value(self):
        p = Param("a", tp=int, value=1)
        assert p.value == 1
        assert p() == 1

    def test_init_with_invalid_value_raise_error(self):
        with pytest.raises(TypeError):
            Param("a", tp=int, value=1.)

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
        # only torch.Tensor can be iterable at this moment
        iter_container = IterableContainer(Container([torch.tensor(1), torch.tensor(2)]), 0)
        p = Param("a", value=iter_container)
        assert isinstance(p.container, Container)
        assert p.container.value is iter_container
        assert p.value == torch.tensor(1)

    def test_get_iterable_input_value(self):
        # only torch.Tensor can be iterable at this moment
        p = Param("a", tp=List[torch.Tensor])
        p.container = IterableInputContainers(IterableContainer(Container(torch.tensor(1)), 1))
        p.container.add(IterableContainer(Container(torch.tensor(2)), 0))
        assert p.value == [torch.tensor(2), torch.tensor(1)]

    def test_select(self):
        p = Param("a", List[torch.Tensor], value=[torch.tensor(1), torch.tensor(1)])
        assert p._is_subscriptable
        assert p.is_subscriptable
        iter_param = p.select(0)
        assert isinstance(iter_param, IterableParam)
        assert isinstance(iter_param.iter_container, IterableContainer)
        assert iter_param.iter_container.index == 0
        assert iter_param.iter_container.value == 1

    def test_subscriptable(self):
        p = Param("a", List[torch.Tensor], value=[torch.tensor(1), torch.tensor(1)])
        assert p.is_subscriptable
        p.set_as_non_subscriptable()
        assert p.is_subscriptable is False
        p.reset_is_subscriptable()
        assert p.is_subscriptable

    def test_connect_iterparam_param_no_select_raise_error(self):
        p0 = Param("a", List[torch.Tensor])
        p1 = Param("b")
        # TODO: this check is temporary disabled because we should allow connect iterparam with iterparam
        # with pytest.raises(ValueError):
        #    # mandatory connect with an index
        #    p0.connect(p1)
        p0.select(0).connect(p1)

    def test_connect_param_iterparam_no_select_raise_error(self):
        p0 = Param("a", tp=List[torch.Tensor])
        p1 = Param("b", value=torch.Tensor(2))
        # TODO: this check is temporary disabled because we should allow connect iterparam with iterparam
        # with pytest.raises(ValueError):
        #    # mandatory connect with an index
        #    p1.connect(p0)
        p1.connect(p0.select(0))

    def test_connect_param_iterparam_no_valid_type_raise_error(self):
        p0 = Param("a", tp=List[torch.Tensor])
        p1 = Param("b", value=2)
        with pytest.raises(TypeError):
            p1.connect(p0.select(0))

    def test_connect_param_param_no_valid_type_raise_error(self):
        p0 = Param("a", tp=int)
        p1 = Param("b", value=2.)
        with pytest.raises(TypeError):
            p1.connect(p0)

    def test_connect_param_param(self):
        p0 = Param("a", value=1)
        p1 = Param("b")
        p0.connect(p1)
        assert isinstance(p1.container, Container)
        assert p1.value == 1
        assert p0.references == {Reference(p1, ori_param=p0)}
        assert list(p0._refs.keys()) == [None]
        assert list(p0._refs[None])[0] == Reference(p1, ori_param=p0)
        assert p1.references == {Reference(p0, ori_param=p1)}
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None])[0] == Reference(p0, ori_param=p1)

    def test_disconnect_param_param(self):
        p0 = Param("a", value=1)
        p1 = Param("b")
        p0.connect(p1)
        p0.disconnect(p1)
        assert isinstance(p1.container, Container)
        assert isinstance(p1.value, NoValue)
        assert p0.references == set()
        assert list(p0._refs.keys()) == [None]
        assert list(p0._refs[None]) == []
        assert p1.references == set()
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None]) == []

    def test_connect_disconnect_iterparam_param(self):
        p0 = Param("a", tp=List[torch.Tensor], value=[torch.tensor(1), torch.tensor(2)])
        p1 = Param("b")
        p0.select(1).connect(p1)
        assert isinstance(p1.container, Container)
        assert p1.value == torch.tensor(2)
        assert list(p0._refs.keys()) == [1]
        assert list(p0._refs[1])[0] == Reference(p1, ori_param=p0, ori_index=1)
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None])[0] == Reference(p0, p1, 1)

    def test_disconnect_iterparam_param(self):
        p0 = Param("a", tp=List[torch.Tensor], value=[torch.tensor(1), torch.tensor(2)])
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
        p0 = Param("a", tp=List[torch.Tensor])
        p1 = Param("b", value=torch.tensor(1))
        p2 = Param("c", value=torch.tensor(2))
        p1.connect(p0.select(1))
        p2.connect(p0.select(0))
        assert isinstance(p0.container, IterableInputContainers)
        assert p0.value == [torch.tensor(2), torch.tensor(1)]
        assert sorted(list(p0._refs.keys())) == sorted([0, 1])
        assert list(p0._refs[0])[0] == Reference(p2, ori_param=p0, ori_index=0)
        assert list(p0._refs[1])[0] == Reference(p1, ori_param=p0, ori_index=1)
        assert list(p1._refs.keys()) == [None]
        assert list(p1._refs[None])[0] == Reference(p0, p1, 1)
        assert list(p2._refs.keys()) == [None]
        assert list(p2._refs[None])[0] == Reference(p0, p2, 0)

    def test_disconnect_param_iterparam(self):
        p0 = Param("a", tp=List[torch.Tensor])
        p1 = Param("b", value=torch.tensor(1))
        p2 = Param("c", value=torch.tensor(2))
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
        p0 = Param("a", tp=List[torch.Tensor], value=[torch.tensor(1), torch.tensor(2)])
        p1 = Param("b", tp=List[torch.Tensor])
        p0.select(0).connect(p1.select(1))
        p0.select(1).connect(p1.select(0))
        assert isinstance(p1.container, IterableInputContainers)
        assert p1.value == [torch.tensor(2), torch.tensor(1)]
        assert sorted(list(p0._refs.keys())) == sorted([0, 1])
        assert list(p0._refs[0])[0] == Reference(p1, p0, 1, 0)
        assert list(p0._refs[1])[0] == Reference(p1, p0, 0, 1)
        assert sorted(list(p1._refs.keys())) == sorted([0, 1])
        assert list(p1._refs[0])[0] == Reference(p0, p1, 1, 0)
        assert list(p1._refs[1])[0] == Reference(p0, p1, 0, 1)

    def test_disconnect_iterparam_iterparam(self):
        p0 = Param("a", tp=List[torch.Tensor], value=[torch.tensor(1), torch.tensor(2)])
        p1 = Param("b", tp=List[torch.Tensor])
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
        p0 = Param("a", List[torch.Tensor], value=[torch.tensor(1), torch.tensor(2)])
        p1 = Param("b")
        p2 = Param("c")
        p0.select(0).connect(p1)
        p0.select(1).connect(p2)
        assert p1.ref_counter() == 1
        assert p2.ref_counter() == 1
        assert p0.ref_counter() == 2
        assert p0.ref_counter(0) == 1
        assert p0.ref_counter(1) == 1


@pytest.mark.usefixtures("event_loop_instance")
class TestIterableParam:
    def test_smoke(self):
        p = Param("a", tp=List[int], value=[1, 2])
        ip = IterableParam(p, 0)
        assert ip.param is p
        assert ip.index == 0
        assert ip.value == 1
        assert isinstance(ip.iter_container, IterableContainer)
        assert ip.iter_container.index == 0
        assert ip.iter_container.value == 1

    def test_ref_counter(self):
        p0 = Param("a", tp=List[int], value=[1, 2])
        ip = IterableParam(p0, 0)
        assert ip.ref_counter() == 0
        p1 = Param("b", tp=int)
        ip.connect(p1)
        assert ip.ref_counter() == 1


def test_check_subscriptable():
    # only torch.Tensor is subscriptable
    assert not limbus.core.param._check_subscriptable(Sequence[int])
    assert not limbus.core.param._check_subscriptable(Iterable[int])
    assert not limbus.core.param._check_subscriptable(List[int])
    assert not limbus.core.param._check_subscriptable(Tuple[int])
    assert limbus.core.param._check_subscriptable(Sequence[torch.Tensor])
    assert limbus.core.param._check_subscriptable(Iterable[torch.Tensor])
    assert limbus.core.param._check_subscriptable(List[torch.Tensor])
    assert limbus.core.param._check_subscriptable(Tuple[torch.Tensor])
    assert not limbus.core.param._check_subscriptable(Tuple[torch.Tensor, torch.Tensor])
    assert not limbus.core.param._check_subscriptable(int)
    assert not limbus.core.param._check_subscriptable(torch.Tensor)
    assert not limbus.core.param._check_subscriptable(str)


class A(Component):
    async def forward(self) -> ComponentState:
        return ComponentState.OK


class B(Component):
    def __init__(self, name):
        super().__init__(name)
        self.stopping_execution = 1

    async def forward(self) -> ComponentState:
        return ComponentState.OK


class TestInputParam:
    def test_smoke(self):
        p = InputParam("a")
        assert isinstance(p, Param)

    async def test_receive_without_parent(self):
        p = InputParam("a")
        with pytest.raises(AssertionError):
            await p.receive()

    async def test_receive_without_refs(self):
        p = InputParam("a", value=1, parent=A("a"))
        assert await p.receive() == 1

    async def test_receive_with_refs(self):
        po = OutputParam("b", parent=A("b"))
        pi = InputParam("a", parent=A("a"))
        po >> pi
        res = await asyncio.gather(po.send(1), pi.receive())
        assert res == [None, 1]
        assert [ref.consumed.is_set() for ref in pi.references] == [True]
        assert [ref.sent.is_set() for ref in po.references] == [False]

    async def test_receive_with_refs_and_stopping_iteration(self):
        po = OutputParam("b", parent=B("b"))
        pi = InputParam("a", parent=B("a"))
        po >> pi
        res = await asyncio.gather(po.send(1), pi.receive())
        assert res == [None, 1]
        assert [ref.consumed.is_set() for ref in pi.references] == [True]
        assert [ref.sent.is_set() for ref in po.references] == [False]
        assert pi.parent.executions_counter == 1

    async def test_receive_from_iterable_param(self):
        po0 = OutputParam("b", torch.Tensor, parent=A("b"))
        po1 = OutputParam("c", torch.Tensor, parent=A("c"))
        pi = InputParam("a", List[torch.Tensor], parent=A("a"))
        po0 >> pi.select(0)
        po1 >> pi.select(1)
        t0 = asyncio.create_task(pi.receive())
        await asyncio.sleep(0)  # exec t0 without blocking
        assert list(pi._refs[0])[0].consumed.is_set() is False
        assert list(pi._refs[1])[0].consumed.is_set() is False
        assert list(pi._refs[0])[0].sent.is_set() is False
        assert list(pi._refs[1])[0].sent.is_set() is False
        t1 = asyncio.create_task(po1.send(torch.tensor(1)))
        await asyncio.sleep(0)  # exec t1 without blocking
        assert list(pi._refs[0])[0].consumed.is_set() is False
        assert list(pi._refs[1])[0].consumed.is_set() is False
        assert list(pi._refs[0])[0].sent.is_set() is False
        assert list(pi._refs[1])[0].sent.is_set() is True
        t2 = asyncio.create_task(po0.send(torch.tensor(2)))
        await asyncio.sleep(0)  # exec t2 without blocking
        assert list(pi._refs[0])[0].consumed.is_set() is False
        assert list(pi._refs[1])[0].consumed.is_set() is False
        assert list(pi._refs[0])[0].sent.is_set() is True
        assert list(pi._refs[1])[0].sent.is_set() is True
        await asyncio.gather(t0, t1, t2)
        assert list(pi._refs[0])[0].consumed.is_set() is True
        assert list(pi._refs[1])[0].consumed.is_set() is True
        assert list(pi._refs[0])[0].sent.is_set() is False
        assert list(pi._refs[1])[0].sent.is_set() is False
        assert pi.value == [torch.tensor(2), torch.tensor(1)]


class TestOutputParam:
    def test_smoke(self):
        p = OutputParam("a")
        assert isinstance(p, Param)

    async def test_send_without_parent(self):
        p = OutputParam("a")
        with pytest.raises(AssertionError):
            await p.send(1)

    async def test_send_without_refs(self):
        p = OutputParam("a", value=1, parent=A("a"))
        await p.send(1)

    async def test_send_with_refs(self):
        po = OutputParam("b", parent=A("b"))
        pi = InputParam("a", parent=A("a"))
        po >> pi
        assert [ref.sent.is_set() for ref in po.references] == [False]
        asyncio.create_task(po.send(1))
        await asyncio.sleep(0)
        assert [ref.sent.is_set() for ref in po.references] == [True]
        assert [ref.consumed.is_set() for ref in po.references] == [False]
        await pi.receive()
        await asyncio.sleep(0)

    async def test_send_from_iterable_param(self):
        po = OutputParam("c", List[torch.Tensor], parent=A("c"))
        pi0 = InputParam("a", torch.Tensor, parent=A("a"))
        pi1 = InputParam("b", torch.Tensor, parent=A("b"))
        po.select(0) >> pi0
        po.select(1) >> pi1
        t0 = asyncio.create_task(pi0.receive())
        await asyncio.sleep(0)  # exec t0 without blocking
        assert list(po._refs[0])[0].consumed.is_set() is False
        assert list(po._refs[1])[0].consumed.is_set() is False
        assert list(po._refs[0])[0].sent.is_set() is False
        assert list(po._refs[1])[0].sent.is_set() is False
        t1 = asyncio.create_task(po.send([torch.tensor(1), torch.tensor(2)]))
        await asyncio.sleep(0)  # exec t1 without blocking
        assert list(po._refs[0])[0].consumed.is_set() is False
        assert list(po._refs[1])[0].consumed.is_set() is False
        assert list(po._refs[0])[0].sent.is_set() is True
        assert list(po._refs[1])[0].sent.is_set() is True
        t2 = asyncio.create_task(pi1.receive())
        await asyncio.gather(t0, t1, t2)
        assert list(po._refs[0])[0].consumed.is_set() is True
        assert list(po._refs[1])[0].consumed.is_set() is True
        assert list(po._refs[0])[0].sent.is_set() is False
        assert list(po._refs[1])[0].sent.is_set() is False
        assert pi0.value == torch.tensor(1)
        assert pi1.value == torch.tensor(2)
