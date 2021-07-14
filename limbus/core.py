from abc import abstractmethod
from typing import Dict, Any, List
from enum import Enum

import torch.nn as nn


class ComponentState(Enum):
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
    def forward(self, inputs: Params, outputs: Params) -> ComponentState:
        return ComponentState.NotImplemented


class ComponentsManager(nn.Module):
    """Class to manage the Limbus Components.
    
    It holds the logic to construct the pipeline to link components.
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]]= {}
        self._seq = []

    def connect(self, _from: Component, _from_name: str, _to: Component, _to_name: str) -> None:
        """Method to connect one component output to another component input.
        
        Args:
            _from: the origin component instance.
            _from_name: the origin component member name.
            _to: the destination component instance.
            _to_name: the destination component member name.
        """
        for t in [_from, _to]:
            if t.name not in self.nodes.keys():
                if not len(t.inputs.__dict__.keys()) and not len(t.outputs.__dict__.keys()):
                    raise ValueError("No inputs and outputs provided!!!")
                inp = {}
                node_type = NodeType.Mid
                for k in t.inputs.__dict__.keys():
                    inp[k] = {}
                if not len(t.inputs.__dict__.keys()):
                    node_type = NodeType.Start
                out = {}
                for k in t.outputs.__dict__.keys():
                    out[k] = {}
                if not len(t.outputs.__dict__.keys()):
                    node_type = NodeType.End
                self.nodes[t.name] = {"type": node_type, "inputs": inp, "outputs": out, "object": t}

        if not self.nodes[_from.name]["outputs"][_from_name]:
            self.nodes[_from.name]["outputs"][_from_name] = {_to.name: input}
        else:
            raise ValueError("Already setted!!!")

        if not self.nodes[_to.name]["inputs"][_to_name]:
            self.nodes[_to.name]["inputs"][_to_name] = {_from.name: _from_name}
        else:
            raise ValueError("Already setted!!!")

    def _traverse(self, k) -> List:
        out = self.nodes[k]["outputs"].values()
        for comp in out:
            k = list(comp.keys())[0]
            vs = self.nodes[k]
            for _, inp in vs["inputs"].items():
                if list(inp.keys())[0] not in self._seq:
                    return
            self._seq.append(k)
            self._traverse(k)

    def traverse(self):
        """Method to traverse the components graph before execution."""
        # find start nodes
        self._seq = []

        for k, vs in self.nodes.items():
            if vs["type"] == NodeType.Start:
                self._seq.append(k)
                self._traverse(k)

    def execute(self):
        """Method to execute the components graph."""
        for mod in self._seq:
            obj = self.nodes[mod]["object"]
            for k, v in self.nodes[mod]["inputs"].items():
                obj.inputs.__dict__[k] = (
                    self.nodes[list(v.keys())[0]]["object"].outputs.__dict__[list(v.values())[0]])
            obj.forward(obj.inputs, obj.outputs)