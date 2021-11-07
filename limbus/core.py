"""Core methods to manage components."""
from abc import abstractmethod
from typing import Callable, Dict, Any, List, Collection, Optional, Tuple, cast, Union, OrderedDict
import torch
import typeguard
from enum import Enum
from dataclasses import dataclass
import logging
import time
import inspect
import collections

import torch.nn as nn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ComponentState(Enum):
    """Possible states for the components."""
    STOPPED = 0
    OK = 1
    NotImplemented = -1
    ERROR = 2
    DISABLED = 3


class NodeType(Enum):
    """Types of nodes."""
    Start = 0
    Mid = 1
    End = 2


class NoValue():
    """Denote that a param does not have a value."""
    pass


class Params:
    """Class to store parameters."""
    def __init__(self) -> None:
        self._types: Dict[str, Any] = {}
        self._args: Dict[str, Optional[str]] = {}

    def declare(self, name: str, tp: Any = Any, value: Any = NoValue(), arg: Optional[str] = None) -> None:
        """Add or modify a param.

        Args:
            name: name of the parameter.
            tp: type (e.g. str, int, list, Union[str, int]...). Default: typing.Any
            value (optional): value for the parameter. Default: NoValue().
            arg (optional): argument directly related with the value of the parameter. Default: None.
                            E.g. this is useful to propagate datatypes and values from the target pin to the arg.

        """
        if not isinstance(value, NoValue):
            typeguard.check_type("value", value, tp)
        setattr(self, name, value)
        self._types[name] = tp
        self._args[name] = arg

    def get_related_arg(self, name: str) -> Optional[str]:
        """Return the argument related with a given param.

        Args:
            name: name of the param.

        """
        return self._args[name]

    def get_params(self) -> Collection[str]:
        """Return the name of all the params."""
        return self._types.keys()

    def get_types(self) -> Dict[str, type]:
        """Return the name and the type of all the params."""
        return self._types

    def get_type(self, name: str) -> type:
        """Return the type of a given param.

        Args:
            name: name of the param.

        """
        return self._types[name]

    def get_param(self, name: str) -> Any:
        """Return the param value after checking the type.

        Args:
            name: name of the param.

        """
        typeguard.check_type(name, getattr(self, name), self.get_type(name))
        return getattr(self, name)

    def set_param(self, name: str, value: Any) -> None:
        """Set the param value after checking the type.

        Args:
            name: name of the param.
            value: value to be setted.

        """
        typeguard.check_type(name, value, self.get_type(name))
        setattr(self, name, value)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __repr__(self) -> str:
        return ''.join(
            (
                f'{type(self).__name__}(',
                ', '.join(
                    f'{name}={getattr(self, name)}' for name in sorted(self.__dict__) if not name.startswith('_')
                ),
                ')',
            )
        )


class Component(nn.Module):
    """Base class to define a Limbus Component."""
    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._inputs = self.__class__.register_inputs()
        self._outputs = self.__class__.register_outputs()

    @staticmethod
    def register_inputs() -> Params:
        """Register the input params.

        Returns:
            input params

        """
        return Params()

    @staticmethod
    def register_outputs() -> Params:
        """Register the output params.

        Returns:
            output params

        """
        return Params()

    @property
    def name(self) -> str:
        """Name of the component."""
        return self._name

    @property
    def inputs(self) -> Params:
        """Get the set of component inputs."""
        return self._inputs

    @property
    def outputs(self) -> Params:
        """Get the set of component outputs."""
        return self._outputs

    @abstractmethod
    def forward(self, inputs: Params) -> ComponentState:
        """Run the component.

        Args:
         inputs: set of values to be used to run the component.

        """
        return ComponentState.NotImplemented

    @abstractmethod
    def finish_iter(self) -> None:
        """Event executed when a pipeline iter is finished."""
        pass


@dataclass
class _Link:
    node: str
    pin: str
    idx: Optional[int] = None  # index of the pin data being used. E.g. tensor[:, idx, :, :]


@dataclass
class _Node:
    type: NodeType
    component: Component
    # inputs and outputs are lists of _Link but we could need to access them through an index
    inputs: Dict[Any, List[Union[Dict[int, _Link], _Link]]]
    outputs: Dict[Any, List[Union[Dict[int, _Link], _Link]]]


class DefaultParam():
    """Trick to denote when to use a default parameter."""
    pass


class ComponentsManager(nn.Module):
    """Class to manage the Limbus Components.

    It holds the logic to construct the pipeline to link components.
    """
    def __init__(self):
        self.nodes: Dict[str, _Node] = {}
        self._seq = []

    def connect(self,
                _from: Component, from_name: Union[str, Tuple[str, int]],
                _to: Component, to_name: Union[str, Tuple[str, int]]) -> None:
        """Connect one component output to another component input.

        Args:
            _from: the origin component instance.
            from_name: the origin component member name. Can be a tuple (name, index) where index denotes the internal
                        element to be connected.
            _to: the destination component instance.
            _to_name: the destination component member name. Can be a tuple (name, index) where index denotes the
                        internal element to be connected.

        """
        from_idx: Optional[int] = from_name[1] if isinstance(from_name, tuple) else None
        from_name = from_name if isinstance(from_name, str) else from_name[0]
        to_idx: Optional[int] = to_name[1] if isinstance(to_name, tuple) else None
        to_name = to_name if isinstance(to_name, str) else to_name[0]
        for comp in [_from, _to]:
            if comp.name not in self.nodes.keys():
                # check type of component and list all the input and output pins
                inp_keys: Collection[Any] = comp.inputs.get_params()
                out_keys: Collection[Any] = comp.outputs.get_params()
                if not len(inp_keys) and not len(out_keys):
                    raise ValueError("No inputs and outputs provided!!!")

                inp: Dict[Any, List] = {key: [] for key in inp_keys}
                out: Dict[Any, List] = {key: [] for key in out_keys}

                node_type = NodeType.Mid
                if not len(inp_keys):
                    node_type = NodeType.Start
                if not len(out_keys):
                    node_type = NodeType.End
                # add the node to the graph
                self.nodes[comp.name] = _Node(node_type, comp, inp, out)

        # set connections between nodes (they can be multiple)
        to_link: Union[Dict[int, _Link], _Link] = _Link(_to.name, to_name, to_idx)
        from_link: Union[Dict[int, _Link], _Link] = _Link(_from.name, from_name, from_idx)
        if from_idx is not None:
            assert isinstance(to_link, _Link)
            to_link = {from_idx: to_link}
        if to_idx is not None:
            assert isinstance(from_link, _Link)
            from_link = {to_idx: from_link}
        self.nodes[_from.name].outputs[from_name].append(to_link)
        self.nodes[_to.name].inputs[to_name].append(from_link)

    def _traverse(self, node_name) -> None:
        # remainder about teh pins:
        # - they can be entirely assigned to another pin (then the type is _Link)
        # - they can be assigned by slices (then the type is Dict[int, _Link])
        out_pins: Dict[Any, List[Union[Dict[int, _Link], _Link]]] = self.nodes[node_name].outputs
        inp_links: List[Union[Dict[int, _Link], _Link]]
        link: Union[Dict[int, _Link], _Link]
        for inp_links in out_pins.values():
            # each output pin can be linked to several input pins
            for link in inp_links:
                if isinstance(link, dict):
                    # in the traverse we do not care about the indexes
                    link = list(link.values())[0]
                next_inp_link: bool = False
                dst_node: _Node = self.nodes[link.node]
                # check if all the components required by the input pins are added
                dst_inp_pin: List[Union[Dict[int, _Link], _Link]]
                for _, dst_inp_pin in dst_node.inputs.items():
                    # check that all the connected nodes for that link are added
                    for out_src_link in dst_inp_pin:
                        if isinstance(out_src_link, dict):
                            # in the traverse we do not care about the indexes
                            out_src_link = list(out_src_link.values())[0]
                        # if not all the source nodes are already in teh seq, we need to add them before
                        if out_src_link.node not in self._seq:
                            # jump to the next link in the list
                            next_inp_link = True
                            break
                    if next_inp_link:
                        break
                if next_inp_link:
                    continue
                # add dst_node id to the list
                if link.node not in self._seq:
                    self._seq.append(link.node)
                # continue the traverse once all the input links are processed
                self._traverse(link.node)

    def traverse(self) -> None:
        """Traverse the components graph before execution."""
        # find start nodes
        self._seq = []

        node_name: str
        node: _Node
        for node_name, node in self.nodes.items():
            if node.type == NodeType.Start:
                self._seq.append(node_name)
                self._traverse(node_name)

    def execute(self, iters: Optional[int] = None) -> None:
        """Execute the components graph.

        Args:
            iters (optional): number of iters to be run. By default all of them are run.

        """
        state = ComponentState.STOPPED
        count = 1
        while True:
            if iters is not None and iters < count:
                break
            log.info(f"Iteration {count}")
            count += 1
            for node_name in self._seq:
                obj = self.nodes[node_name].component
                pin: str
                links: List[Union[Dict[int, _Link], _Link]]
                # get values from the connected components
                inputs = Params()
                for pin, links in self.nodes[node_name].inputs.items():
                    # get input value for each pin
                    # each pin can have one value except in case of slices
                    values: Union[OrderedDict[int, Any], Any] = collections.OrderedDict()
                    for link in links:
                        if len(links) > 1:
                            # if we have more than one link per pin they must be slices
                            try:
                                typeguard.check_type("",link, Dict[int, _Link])
                            except:
                                raise ValueError(f"There are several links connected to the input pin '{pin}' in "
                                                 f"'{obj.__repr__()}'. Input pins do not accept multiple assignements.")

                        idx: Optional[int] = None
                        try:
                            # trick to use typeguard to check the type. If it is a dict then idx and link are updated
                            typeguard.check_type("", link, Dict[int, _Link])
                            assert isinstance(link, dict)
                            idx = list(link.keys())[0]
                            link = list(link.values())[0]
                        except TypeError:
                            pass

                        assert isinstance(link, _Link)
                        value = self.nodes[link.node].component.outputs[link.pin]
                        if link.idx is not None:
                            # select the correct value from the list
                            value = value[link.idx]

                        if idx is not None:
                            if idx in values:
                                raise ValueError(f"There are several links connected to the input pin '{pin}' in "
                                                 f"'{obj.__repr__()}'. Input pins do not accept multiple assignements.")
                            values[idx] = value
                        else:
                            # if it is not a slice we can assign the value directly
                            if isinstance(value, DefaultParam):
                                # without slicing we accept default values
                                values = obj.inputs[pin].default
                            else:
                                values = value
                            break

                    if not isinstance(values, OrderedDict):
                        inputs.declare(pin, obj.inputs.get_type(pin), values)
                    else:
                        # create list with values sorted according to the idx
                        # NOTE the idx do not need to be consecutive!!!!
                        # NOTE new_values type should be Sequence[torch.Tensor]
                        new_values = []
                        for k in sorted(values.keys()):
                            # assert isinstance(values[k], torch.Tensor)
                            new_values.append(values[k])
                        inputs.declare(pin, obj.inputs.get_type(pin), new_values)
                state = obj.forward(inputs)

                if state == ComponentState.STOPPED:
                    log.info(f"Component {obj.name} stopped the pipeline.")
                    break
                if state == ComponentState.DISABLED:
                    log.warning(f"Component {obj.name} is DISABLED.")
                if state == ComponentState.DISABLED:
                    log.error(f"Component {obj.name} produced an ERROR.")
                    break

            # code to run before running the next iteration in the pipeline
            for node_name in self._seq:
                self.nodes[node_name].component.finish_iter()
            if state == ComponentState.STOPPED:
                log.info("DONE")
                break
            time.sleep(2)


def component_factory(callable_to_wrap: Callable) -> Component:
    """Generate a Component class for a given callable.

    Args:
        callable_to_wrap: callable to be wrapped as a component.

    Returns:
        Component wrapping the callable.

    """
    if not inspect.isfunction(callable_to_wrap) and not inspect.isbuiltin(callable_to_wrap):
        raise TypeError(f"{callable_to_wrap} is not a function. At this moment only functio callables are supported.")

    # overwrite the forward(), register_inputs() and register_outputs() methods.
    def forward(self, inputs: Params) -> ComponentState:  # noqa: D417
        """Run the component.

        Args:
            inputs: set of values to be used to run the component.

        """
        args: Dict[str, Any] = {}
        for param in self._inputs.get_params():
            args[param] = inputs.get_param(param)

        res = callable_to_wrap(**args)

        if len(self._outputs.get_params()) > 1:
            for idx, param in enumerate(self._outputs.get_params()):
                self._outputs.set_param(param, res[idx])
        else:
            param = list(self._outputs.get_params())[0]
            self._outputs.set_param(param, res)
        return ComponentState.OK

    def register_inputs() -> Params:
        """Register the inputs params.

        Returns:
            input params

        """
        inputs = Params()
        sign: inspect.Signature = inspect.signature(callable_to_wrap)
        for param in sign.parameters.values():
            if param.default is param.empty:
                inputs.declare(param.name, param.annotation)
            else:
                inputs.declare(param.name, param.annotation, param.default)
        return inputs

    def register_outputs() -> Params:
        """Register the output params.

        Returns:
            output params

        """
        def isinstance_namedtuple(obj) -> bool:
            if typeguard.isclass(obj):
                return issubclass(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')
            return False
        outputs = Params()
        sign: inspect.Signature = inspect.signature(callable_to_wrap)
        if isinstance_namedtuple(sign.return_annotation):
            # variable number of outputs
            for k, v in sign.return_annotation._field_defaults.items():
                outputs.declare(k, v)
        else:
            if not typeguard.isclass(sign.return_annotation) and sign.return_annotation._name == "Tuple":
                # variable number of outputs
                for idx, arg in enumerate(sign.return_annotation.__args__):
                    outputs.declare(f"out{idx}", arg)
            else:
                # single output case
                outputs.declare("out", sign.return_annotation)
        return outputs

    return cast(Component, type(
        callable_to_wrap.__name__, (Component,),
        {"forward": forward, "register_inputs": register_inputs, "register_outputs": register_outputs}
        )
    )
