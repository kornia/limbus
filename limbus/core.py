"""Core methods to manage components."""
from abc import abstractmethod
from typing import Dict, Any, List, Collection, Optional, Tuple
from typeguard import check_type
from enum import Enum
from dataclasses import dataclass
import logging
import time

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
            check_type("value", value, tp)
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
        check_type(name, getattr(self, name), self.get_type(name))
        return getattr(self, name)

    def set_param(self, name: str, value: Any) -> None:
        """Set the param value after checking the type.

        Args:
            name: name of the param.
            value: value to be setted.

        """
        check_type(name, value, self.get_type(name))
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
        (self._inputs, self._outputs) = self.define_params()

    @classmethod
    def define_params(cls) -> Tuple[Params, Params]:
        """Define the input and output params without instantiating the class.

        Returns:
            (input params, output params)

        """
        return (Params(), Params())

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


@dataclass
class _Node:
    type: NodeType
    component: Component
    inputs: Dict[Any, List[_Link]]
    outputs: Dict[Any, List[_Link]]


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

    def connect(self, _from: Component, _from_name: str, _to: Component, _to_name: str) -> None:
        """Connect one component output to another component input.

        Args:
            _from: the origin component instance.
            _from_name: the origin component member name.
            _to: the destination component instance.
            _to_name: the destination component member name.

        """
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
        self.nodes[_from.name].outputs[_from_name].append(_Link(_to.name, _to_name))
        self.nodes[_to.name].inputs[_to_name].append(_Link(_from.name, _from_name))

    def _traverse(self, node_name) -> None:
        out_pins: Dict[Any, List[_Link]] = self.nodes[node_name].outputs
        inp_links: List[_Link]
        link: _Link
        for inp_links in out_pins.values():
            # each output pin can be linked to several input pins
            for link in inp_links:
                next_inp_link: bool = False
                dst_node: _Node = self.nodes[link.node]
                # check if all the components required by the input pins are added
                dst_inp_pin: List[_Link]
                for _, dst_inp_pin in dst_node.inputs.items():
                    # check that all the connected nodes for that link are added
                    for out_src_link in dst_inp_pin:
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
                links: List[_Link]
                # get values from the connected components
                inputs = Params()
                for pin, links in self.nodes[node_name].inputs.items():
                    # get input value for each pin
                    values = []  # for now we assume that params are passed as a list
                    for link in links:
                        value = self.nodes[link.node].component.outputs[link.pin]
                        if isinstance(value, DefaultParam):
                            values.append(obj.inputs[pin])
                        else:
                            values.append(value)
                    if len(links) == 1:
                        inputs.declare(pin, obj.inputs.get_type(pin), values[0])
                    else:
                        inputs.declare(pin, obj.inputs.get_type(pin), values)
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
