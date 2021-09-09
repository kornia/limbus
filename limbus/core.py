from abc import abstractmethod
from typing import Dict, Any, List, Collection
from enum import Enum
from dataclasses import dataclass
import logging

import torch.nn as nn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ComponentState(Enum):
    STOPPED = 0
    OK = 1
    NotImplemented = -1


class NodeType(Enum):
    Start = 0
    Mid = 1
    End = 2


class Params:
    """Class to store parameters."""
    def declare(self, name: str, value: Any = None):
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
        self._inputs = Params()
        self._outputs = Params()

    @property
    def name(self) -> str:
        return self._name

    @property
    def inputs(self) -> Params:
        return self._inputs

    @property
    def outputs(self) -> Params:
        return self._outputs

    @abstractmethod
    def forward(self, inputs: Params) -> ComponentState:
        return ComponentState.NotImplemented

    @abstractmethod
    def finish_iter(self) -> None:
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
                inp_keys: Collection[Any] = comp.inputs.__dict__.keys()
                out_keys: Collection[Any] = comp.outputs.__dict__.keys()
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
        """Method to traverse the components graph before execution."""
        # find start nodes
        self._seq = []

        node_name: str
        node: _Node
        for node_name, node in self.nodes.items():
            if node.type == NodeType.Start:
                self._seq.append(node_name)
                self._traverse(node_name)

    def execute(self) -> None:
        """Execute the components graph."""
        state = ComponentState.STOPPED
        count = 1
        while True:
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
                        value: Any = self.nodes[link.node].component.outputs[link.pin]
                        if isinstance(value, DefaultParam):
                            values.append(obj.inputs[pin])
                        else:
                            values.append(value)
                    if len(links) == 1:
                        inputs.declare(pin, values[0])
                    else:
                        inputs.declare(pin, values)
                state = obj.forward(inputs)

                if state == ComponentState.STOPPED:
                    break

            # code to run before running the next iteration in the pipeline
            for node_name in self._seq:
                self.nodes[node_name].component.finish_iter()
            if state == ComponentState.STOPPED:
                log.info("DONE")
                break
