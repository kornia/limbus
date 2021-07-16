from abc import abstractmethod
from typing import Dict, Any, List, Collection
from enum import Enum
from dataclasses import dataclass

import torch.nn as nn


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


@dataclass
class _Node:
    type: NodeType
    component: Component
    inputs: Dict[Any, List]
    outputs: Dict[Any, List]


@dataclass
class _Link:
    node: str
    pin: str


class ComponentsManager(nn.Module):
    """Class to manage the Limbus Components.

    It holds the logic to construct the pipeline to link components.
    """
    def __init__(self):
        self.nodes: Dict[str, _Node]= {}
        self._seq = []

    def connect(self, _from: Component, _from_name: str, _to: Component, _to_name: str) -> None:
        """Method to connect one component output to another component input.

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
        pins: Collection[List[_Link]] = self.nodes[node_name].outputs.values()
        pin: List[_Link]
        link: _Link
        for pin in pins:
            for link in pin:
                node: _Node = self.nodes[link.node]
                # check if all the components required by the input pins are added
                inp_links: List[_Link]
                for _, inp_links in node.inputs.items():
                    # check that all the connections for that link are added
                    for inp_link in inp_links:
                        if inp_link.node not in self._seq:
                            return
                self._seq.append(link.node)
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
        """Method to execute the components graph."""
        while True:
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
                        values.append(self.nodes[link.node].component.outputs[link.pin])
                    if len(links) == 1:
                        inputs.declare(pin, values[0])
                    else:
                        inputs.declare(pin, values)
                state = obj.forward(inputs)

                if state == ComponentState.STOPPED:
                    break
