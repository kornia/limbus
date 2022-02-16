"""Components manager to connect, traverse and execute pipelines."""
from dataclasses import dataclass
from typing import Dict, Any, List, Collection, Optional, Tuple, Union, OrderedDict
from enum import Enum
import logging
import collections
import time

import typeguard
import torch.nn as nn

from limbus.core import ComponentState, Component, Params


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes."""
    Start = 0
    Mid = 1
    End = 2


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
            to_name: the destination component member name. Can be a tuple (name, index) where index denotes the
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
        # remainder about the pins:
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
                        # if not all the source nodes are already in the seq, we need to add them before
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
                                typeguard.check_type("", link, Dict[int, _Link])
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
                        value = self.nodes[link.node].component.outputs[link.pin].value
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
