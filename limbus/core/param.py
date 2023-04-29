"""Classes to define parameters."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import typing
from typing import Any, TYPE_CHECKING, Callable
import inspect
import collections
import asyncio
import contextlib
from abc import ABC

import typeguard

from limbus.core.states import ComponentState, ComponentStoppedError
from limbus.core import async_utils
# Note that Component class cannot be imported to avoid circular dependencies.
if TYPE_CHECKING:
    from limbus.core.component import Component

SUBSCRIPTABLE_TYPES: list[type] = []
try:
    import torch
    SUBSCRIPTABLE_TYPES.append(torch.Tensor)
except ImportError:
    pass

try:
    import numpy as np
    SUBSCRIPTABLE_TYPES.append(np.ndarray)
except ImportError:
    pass


class NoValue:
    """Denote that a param does not have a value."""
    pass


@dataclass
class Container:
    """Denote that a param has a value."""
    value: Any


@dataclass
class IterableContainer:
    """Denote that a param has an indexed value.

    Note: In our use case the maximum number of nested IterableContainers is 2.
    This number is not explicitly controlled. It is implicitly controlled in the Param class.

    """
    container: Container | "IterableContainer"
    index: int

    @property
    def value(self) -> Any:
        """Get the value of the container."""
        if isinstance(self.container, Container):
            # return the value of the container at the index
            return self.container.value[self.index]
        else:
            # look for the container value.
            # If it is an IterableContainer means that the final value is nested.
            assert isinstance(self.container, IterableContainer)
            return self.container.value


class IterableInputContainers:
    """Denote that an input param is a sequence of Containers."""
    def __init__(self, container: None | IterableContainer = None):
        containers = []
        if container is not None:
            containers = [container]
        self._containers: list[IterableContainer] = containers

    def __len__(self) -> int:
        return len(self._containers)

    def add(self, container: IterableContainer) -> None:
        """Add an IterableValue to the list of values."""
        self._containers.append(container)

    def remove(self, index: int) -> None:
        """Remove an IterableValue from the list of values."""
        for container in self._containers:
            if container.index == index:
                self._containers.remove(container)
                return

    def get_ordered(self) -> list[Any]:
        """Return a list with the values in the order denoted by the index in the IterableValue."""
        indices: list[int] = []
        for container in self._containers:
            assert isinstance(container, IterableContainer)
            indices.append(container.index)

        containers: list[Any] = []
        for pos_idx in sorted(range(len(indices)), key=indices.__getitem__):  # argsort
            obj: Container | IterableContainer = self._containers[pos_idx].container
            if isinstance(obj, IterableContainer):
                obj = obj.container.value[obj.index]  # type: ignore  # Iterable[Any] is not indexable [index]
            else:
                assert isinstance(obj, Container)
                obj = obj.value
            containers.append(obj)
        return containers


def _check_subscriptable(datatype: type) -> bool:
    """Checf if datatype is subscriptable with tensors inside.

    Args:
        datatype (type): type to be analised.

    Returns:
        bool: True if datatype is a subscriptable with tensors, False otherwise.

    """
    # we need to know if it is a variable size datatype, we assume that all the sequences are variable size
    # if they contain tensors. E.g. list[Tensor], tuple[Tensor], Sequence[Tensor].
    # Note that e.g. for the case tuple[Tensor, Tensor] we don't assume it is variable since the size is known.
    origin = typing.get_origin(datatype)
    if origin is None:  # discard datatypes that are not typing expressions
        return False
    datatype_args: tuple = typing.get_args(datatype)
    if inspect.isclass(origin):
        is_abstract: bool = inspect.isabstract(origin)
        is_abstract_seq: bool = origin is collections.abc.Sequence or origin is collections.abc.Iterable
        # mypy complaints in the case origin is NoneType
        if is_abstract_seq or (not is_abstract and isinstance(origin(), typing.Iterable)):  # type: ignore
            if (len(datatype_args) == 1 or (len(datatype_args) == 2 and Ellipsis in datatype_args)):
                if datatype_args[0] in SUBSCRIPTABLE_TYPES:
                    return True
    return False


class IterableParam:
    """Temporal class to manage indexing inside a parameter."""
    def __init__(self, param: "Param", index: int) -> None:
        self._param: Param = param
        # TODO: validate that _iter_container can be an IterableInputContainers, I feel it cannot!!
        self._iter_container: IterableContainer | IterableInputContainers
        if isinstance(param.container, Container):
            self._iter_container = IterableContainer(param.container, index)
        elif isinstance(param.container, IterableInputContainers):
            # since it is an input, the pointer to the value is not relevant at this stage
            self._iter_container = IterableContainer(Container(None), index)

    @property
    def param(self) -> "Param":
        """Return the base parameter."""
        return self._param

    @property
    def index(self) -> int:
        """Return the selected index in the sequence."""
        if isinstance(self._iter_container, IterableInputContainers):
            raise TypeError("Cannot get the index of a list of input containers.")
        return self._iter_container.index

    @property
    def value(self) -> Any | list[Any]:
        """Get the value of the parameter.

        It can be a list of values if the parameter is an IterableInputContainers.

        """
        if isinstance(self._iter_container, IterableContainer):
            return self._iter_container.value
        else:
            assert isinstance(self._iter_container, IterableInputContainers)
            return self._iter_container.get_ordered()

    @property
    def iter_container(self) -> IterableContainer | IterableInputContainers:
        """Get the container of the parameter."""
        return self._iter_container

    def ref_counter(self) -> int:
        """Return the number of references for this parameter."""
        if isinstance(self._iter_container, IterableInputContainers):
            raise TypeError("At this moment the number of references for IterableInputContainers cannot be retrieved.")
        return self._param.ref_counter(self._iter_container.index)

    def connect(self, dst: "Param" | "IterableParam") -> None:
        """Connect this parameter (output) with the dst (input) parameter."""
        self._param._connect(self, dst)

    def __rshift__(self, rvalue: "Param" | "IterableParam"):
        """Allow to connect params using the >> operator."""
        self.connect(rvalue)

    def disconnect(self, dst: "Param" | "IterableParam") -> None:
        """Disconnect this parameter (output) with the dst (input) parameter."""
        self._param._disconnect(self, dst)


@dataclass
class Reference:
    """Reference to a parameter.

    It is used to keep track of the references to a parameter.

    """
    param: "Param"
    ori_param: "Param"  # added to avoid duplicated references, it is rare but it could happen.
    index: None | int = None
    ori_index: None | int = None  # added to avoid duplicated references, it is rare but it could happen.
    # allow to know if there is a new value for the parameter
    sent: None | asyncio.Event = None
    # allow to know if the value has been consumed
    consumed: None | asyncio.Event = None

    def __hash__(self) -> int:
        # this method is required to be able to use Reference in a set.
        # Note that we don't use the consumed attribute in the hash since it is dynamic.
        return hash((self.param, self.index, self.ori_param, self.ori_index))

    def __eq__(self, other: Any) -> bool:
        # this method is required to be able to use Reference in a set.
        # Note that we don't use the consumed attribute in the hash since it is dynamic.
        if isinstance(other, Reference):
            return (self.param == other.param and self.index == other.index and
                    self.ori_param == other.ori_param and self.ori_index == other.ori_index)
        return False


class Param(ABC):
    """Class to store data for each parameter.

    Args:
        name: name of the parameter.
        tp (optional): type of the parameter. Madnatory for subscriptable params. Default: Any.
        value (optional): value of the parameter. Default: NoValue().
        arg (optional): name of the argument in the component constructor related with this param. Default: None.
        parent (optional): parent component. Default: None.
        callback (optional): callback to be called when the value of the parameter changes. Default: None.

    """
    def __init__(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: None | str = None,
                 parent: None | Component = None, callback: Callable | None = None) -> None:
        # validate that the type is coherent with the value
        if not isinstance(value, NoValue):
            typeguard.check_type(name, value, tp)

        self._name: str = name
        self._type: Any = tp
        self._arg: None | str = arg
        # We store all the references for each param.
        # The key is the slicing for the current param.
        self._refs: dict[Any, set[Reference]] = defaultdict(set)
        self._value: Container | IterableContainer | IterableInputContainers = Container(value)
        # only sequences with tensors inside are subscriptable
        self._is_subscriptable = _check_subscriptable(tp)
        self._parent: None | Component = parent
        self._callback: None | Callable = callback

    @property
    def is_subscriptable(self) -> bool:
        """Return if the parameter is subscriptable."""
        return self._is_subscriptable

    def reset_is_subscriptable(self) -> None:
        """Reset the subscriptable flag."""
        self._is_subscriptable = _check_subscriptable(self._type)

    def set_as_non_subscriptable(self) -> None:
        """Set the subscriptable flag to False."""
        self._is_subscriptable = False

    @property
    def parent(self) -> None | Component:
        """Get the parent component."""
        return self._parent

    @property
    def arg(self) -> None | str:
        """Get the argument related with the param."""
        return self._arg

    @property
    def type(self) -> Any:
        """Return the type of the parameter."""
        return self._type

    @property
    def name(self) -> str:
        """Get the name of the parameter."""
        return self._name

    @property
    def references(self) -> set[Reference]:
        """Get all the references for the parameter."""
        refs: set[Reference] = set()
        for ref_set in self._refs.values():
            refs = refs.union(ref_set)
        return refs

    def __call__(self) -> Any:
        """Get the value of the parameter."""
        return self.value

    @property
    def value(self) -> Any:
        """Get the value of the parameter."""
        if isinstance(self._value, Container):
            if isinstance(self._value.value, IterableContainer):
                # mypy error: Iterable[Any] is not indexable [index]
                return self._value.value.container.value[self._value.value.index]  # type: ignore
            else:
                return self._value.value
        elif isinstance(self._value, IterableInputContainers):
            assert self._is_subscriptable
            origin = typing.get_origin(self._type)
            assert origin is not None
            res_value: list[Any] = self._value.get_ordered()
            return origin(res_value)

    @value.setter
    def value(self, value: Any) -> None:
        """Set the value of the parameter.

        Args:
            value (Any): The value to set.

        """
        self._set_value(value)

    def _set_value(self, value: Any) -> None:
        # trick to easily override the setter of the value property
        if isinstance(value, Param):
            value = value.value
        if not isinstance(self._value, Container):
            raise TypeError(f"Param '{self.name}' cannot be assigned.")
        if isinstance(value, (Container, IterableContainer, set)):
            raise TypeError(
                f"The type of the value to be assigned to param '{self.name}' cannot have a 'value' attribute.")
        typeguard.check_type(self._name, value, self._type)
        self._value.value = value

    @property
    def container(self) -> Container | IterableContainer | IterableInputContainers:
        """Get the container for this parameter."""
        return self._value

    @container.setter
    def container(self, value: Container | IterableContainer | IterableInputContainers) -> None:
        """Set the container for this parameter.

        Args:
            value (Container, IterableContainer or IterableInputContainers): The container to set.

        """
        self._value = value

    def ref_counter(self, index: None | int = None) -> int:
        """Return the number of references for this parameter."""
        if index is not None:
            return len(self._refs[index])
        else:
            return len(self.references)

    def select(self, index: int) -> IterableParam:
        """Select a slice of the parameter.

        Args:
            index (int): The index of the slice.

        Returns:
            Param: The selected slice.

        """
        if not self._is_subscriptable:
            raise ValueError(f"The param '{self.name}' is not subscriptable (it must be a sequence of tensors).")
        # NOTE: we cannot check if the index is valid because it is not known at this point the len of the sequence
        # create a new param with the selected slice inside the param
        return IterableParam(self, index)

    def _connect(self, ori: "Param" | IterableParam, dst: "Param" | IterableParam) -> None:
        """Connect this parameter (output) with the dst (input) parameter."""
        # Disable this check until a better solution is found to connect 2 lists.
        # if isinstance(ori, Param) and ori._is_subscriptable:
        #    raise ValueError(f"The param '{ori.name}' must be connected using indexes.")

        # if isinstance(dst, Param) and dst._is_subscriptable:
        #    raise ValueError(f"The param '{dst.name}' must be connected using indexes.")

        # NOTE that there is not type validation, we will trust in the user to connect params.
        # We only check when there is an explicit value in the ori param.
        if isinstance(ori, Param) and not isinstance(ori.value, NoValue):
            if isinstance(dst, Param):
                typeguard.check_type(self._name, ori.value, dst.type)
            else:
                typeguard.check_type(self._name, ori.value, typing.get_args(dst.param.type)[0])

        # TODO: check that dst param is an input param
        # TODO: check type compatibility
        if (isinstance(dst, Param) and dst.ref_counter() > 0):
            raise ValueError(f"An input parameter can only be connected to 1 param. "
                             f"Dst param '{dst.name}' is connected to {dst._refs}.")

        if isinstance(dst, IterableParam) and dst.param.ref_counter(dst.index) > 0:
            raise ValueError(f"An input parameter can only be connected to 1 param. "
                             f"Dst param '{dst.param.name}' is connected to {dst.param._refs}.")

        # connect the param to the dst param
        if isinstance(dst, Param) and isinstance(ori, Param):
            assert isinstance(dst.container, Container)
            assert isinstance(ori.container, Container)
            dst.container = ori.container
        elif isinstance(dst, IterableParam) and isinstance(ori, Param):
            assert isinstance(dst.iter_container, IterableContainer)
            assert isinstance(ori.container, Container)
            dst.iter_container.container = ori.container
        elif isinstance(dst, Param) and isinstance(ori, IterableParam):
            assert isinstance(dst.container, Container)
            assert isinstance(ori.iter_container, IterableContainer)
            dst.container.value = ori.iter_container
        else:
            assert isinstance(dst, IterableParam)
            assert isinstance(ori, IterableParam)
            assert isinstance(dst.iter_container, IterableContainer)
            assert isinstance(ori.iter_container, IterableContainer)
            dst.iter_container.container = ori.iter_container

        # if dest is an IterableParam means that several ori params can be connected to different dest indexes
        # so they are stored as a list of params
        if isinstance(dst, IterableParam):
            assert isinstance(dst.iter_container, IterableContainer)
            if isinstance(dst.param.container, IterableInputContainers):
                dst.param.container.add(dst.iter_container)
            else:
                dst.param.container = IterableInputContainers(dst.iter_container)

        self._update_references('add', ori, dst)

    def connect(self, dst: "Param" | IterableParam) -> None:
        """Connect this parameter (output) with the dst (input) parameter."""
        self._connect(self, dst)

    def __rshift__(self, rvalue: "Param" | IterableParam):
        """Allow to connect params using the >> operator."""
        self.connect(rvalue)

    def _disconnect(self, ori: "Param" | IterableParam, dst: "Param" | IterableParam) -> None:
        """Disconnect this parameter from the dst parameter."""
        if isinstance(dst, Param):
            assert isinstance(dst.container, Container)
            dst.container = Container(NoValue())
        elif isinstance(dst, IterableParam):
            if isinstance(dst.param.container, IterableInputContainers):
                assert isinstance(dst.iter_container, IterableContainer)
                dst.param.container.remove(dst.iter_container.index)
                if len(dst.param.container) == 0:
                    dst.param.container = Container(NoValue())
            else:
                dst.param.container = Container(NoValue())

        self._update_references('remove', ori, dst)

    def _update_references(self, type: str, ori: "Param" | IterableParam, dst: "Param" | IterableParam
                           ) -> None:
        # assign references
        ori_idx = None
        dst_idx = None
        if isinstance(ori, IterableParam):
            ori_idx = ori.index
            ori = ori.param
        if isinstance(dst, IterableParam):
            dst_idx = dst.index
            dst = dst.param
        if type == 'add':
            # Set events denoting that the param is sent/consumed. Note that the same events are set in the
            # references of both params.
            consumed_event = asyncio.Event()
            sent_event = asyncio.Event()
            ori._refs[ori_idx].add(Reference(dst, ori, dst_idx, ori_idx, sent_event, consumed_event))
            dst._refs[dst_idx].add(Reference(ori, dst, ori_idx, dst_idx, sent_event, consumed_event))
        elif type == 'remove':
            ori._refs[ori_idx].remove(Reference(dst, ori, dst_idx, ori_idx))
            dst._refs[dst_idx].remove(Reference(ori, dst, ori_idx, dst_idx))

    def disconnect(self, dst: "Param" | IterableParam) -> None:
        """Disconnect this parameter (output) from the dst (input) parameter."""
        self._disconnect(self, dst)


class PropParam(Param):
    """Class to manage the comunication for each property parameter."""

    async def set_property(self, value: Any) -> None:
        """Set the value of the property."""
        assert self._parent is not None
        if self._callback is None:
            self.value = value
        else:
            self.value = await self._callback(self._parent, self.value)


class InputParam(Param):
    """Class to manage the comunication for each input parameter."""

    async def receive(self) -> Any:
        """Wait until the input param receives a value from the connected output param."""
        assert self._parent is not None
        self._parent._Component__num_params_waiting_to_receive += 1
        if self.references:
            for ref in self.references:
                # NOTE: each input param can be connected to 0 or 1 output param (N output params if it is iterable).
                # ref: Reference = next(iter(self.references))
                # ensure the component related with the output param exists
                assert ref.param is not None
                ori_param: Param = ref.param
                assert isinstance(ori_param, OutputParam)  # they must be of type OutputParam
                assert ori_param.parent is not None
                self._parent.set_state(ComponentState.RECEIVING_PARAMS,
                                       f"{ori_param.parent.name}.{ori_param.name} -> {self._parent.name}.{self.name}")
                async_utils.create_task_if_needed(self._parent, ori_param.parent)

            if self._parent.stopping_execution == 0:
                # fast way, in contrast with the while loop below, to wait for the input param.
                # wait until all the output params send the values
                await asyncio.gather(*[ref.sent.wait() for ref in self.references if ref.sent is not None])
            else:
                sent: int = 0
                while sent < len(self.references):
                    for ref in self.references:
                        # Trick to avoid issues due to setting a concrete number of iters to be executed. E.g.: if at
                        # least 1 iter is requested from each component and there is a component requesting 2 iters
                        # from a previous one this trick will recreate the tasks.
                        # This is mainly useful for debugging purposes since it slowdown the execution.
                        assert ref.param is not None
                        assert ref.param.parent is not None
                        async_utils.create_task_if_needed(self._parent, ref.param.parent)
                        assert isinstance(ref.sent, asyncio.Event)
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(ref.sent.wait(), timeout=0.1)
                    sent = sum([ref.sent.is_set() for ref in self.references if ref.sent is not None])

            for ref in self.references:
                assert ref.param is not None
                assert ref.param.parent is not None
                # if we want to stop at a given min iter then it is posible to require more iters
                if ComponentState.STOPPED_AT_ITER not in ref.param.parent.state and ref.param.parent.is_stopped():
                    raise ComponentStoppedError(ComponentState.STOPPED_BY_COMPONENT)

            for ref in self.references:
                # NOTE: depending on how the value is consumed we should apply a copy here.
                # - We assume components do not modify the value. (this can happen)
                # - When the value is setted reusing the same memory, instead of creating a new var, then
                # the changes will also be propagated to components consuming the previous value. (in theory
                # this cannot happen)
                # TODO: add a flag to allow to determine if we want to copy the value.
                value = self.value  # get the value before allowing to send again
                assert isinstance(ref.sent, asyncio.Event)
                assert isinstance(ref.consumed, asyncio.Event)
                ref.consumed.set()  # denote that the param is consumed
                ref.sent.clear()  # allow to know to the sender that it can send again
        else:
            value = self.value
        await self._are_all_waiting_params_received()
        if self._callback is not None:
            # specific callback for this param
            value = await self._callback(self._parent, value)

        if self._parent.pipeline and self._parent.pipeline.param_received_user_hook:
            # hook from the pipeline, all the components and input params run the same code
            await self._parent.pipeline.param_received_user_hook(self)
        return value

    async def _are_all_waiting_params_received(self) -> None:
        """Check if the component is waiting for other params before changing the component state."""
        assert self._parent is not None
        self._parent._Component__num_params_waiting_to_receive -= 1
        if self._parent._Component__num_params_waiting_to_receive == 0:
            self._parent.set_state(ComponentState.RUNNING)


class OutputParam(Param):
    """Class to manage the comunication for each output parameter."""

    async def send(self, value: Any) -> None:
        """Send the value of this param to the connected input params."""
        assert self._parent is not None
        if self._callback is None:
            self.value = value  # set the value for the param
        else:
            self.value = await self._callback(self._parent, value)

        for ref in self.references:
            assert isinstance(ref.sent, asyncio.Event)
            assert isinstance(ref.consumed, asyncio.Event)
            ref.consumed.clear()  # init the state of the event
            ref.sent.set()  # denote that the param is ready to be consumed

            # ensure the component related with the input param exists
            assert ref.param is not None
            dst_param: Param = ref.param
            assert isinstance(dst_param, InputParam)  # they must be of type InputParam
            assert dst_param.parent is not None
            self._parent.set_state(ComponentState.SENDING_PARAMS,
                                   f"{self._parent.name}.{self.name} -> {dst_param.parent.name}.{dst_param.name}")
            async_utils.create_task_if_needed(self._parent, dst_param.parent)

        if self._parent.pipeline and self._parent.pipeline.param_sent_user_hook:
            await self._parent.pipeline.param_sent_user_hook(self)

        # wait until all the input params read the value
        await asyncio.gather(*[ref.consumed.wait() for ref in self.references if ref.consumed is not None])
        for ref in self.references:
            assert ref.param is not None
            assert ref.param.parent is not None
            # if we want to stop at a given min iter then it is posible to require more iters
            if ComponentState.STOPPED_AT_ITER not in ref.param.parent.state and ref.param.parent.is_stopped():
                raise ComponentStoppedError(ComponentState.STOPPED_BY_COMPONENT)
