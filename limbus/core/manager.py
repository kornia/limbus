"""Components manager to connect, traverse and execute pipelines."""
from typing import List, Optional, Union, Set
import logging
import time

import typeguard
import torch.nn as nn

from limbus.core import ComponentState, Component, Params, Param


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# TODO: check why I added this DefaultParam class, now it is not used
class DefaultParam():
    """Trick to denote when to use a default parameter."""
    pass


class ComponentsManager(nn.Module):
    """Class to manage the Limbus Components.

    It holds the logic to construct the pipeline to link components.
    """
    def __init__(self):
        self.nodes: Set[Component] = set()
        self._seq: List[Component] = []

    def add(self, components: Union[Component, List[Component]]) -> None:
        """Add components to the manager.

        Args:
            components: Component or list of components to be added.

        """
        if isinstance(components, Component):
            components = [components]
        for component in components:
            self.nodes.add(component)

    def _all_connected(self, ori: Params, dst: Set[Param]) -> bool:
        for p in ori:
            assert len(p._refs) < 2
            # an input param can be only connected to one output param
            if p.ref_counter == 0 or p._refs[0] not in dst:
                return False
        return True

    def _traverse(self, component, traversed_out_params: Set[Param]) -> None:
        # add params of the component since it was already added to the sequence
        for out_param in component.outputs:
            traversed_out_params.add(out_param)

        # search for connected components
        for out_param in component.outputs:
            # search for the references of each output param across all the component
            for in_param in out_param._refs:
                for cmp in self.nodes:
                    if cmp not in self._seq:
                        # found component with all its inputs connected
                        if in_param in cmp.inputs and self._all_connected(cmp.inputs, traversed_out_params):
                            # add component to the execution sequence
                            self._seq.append(cmp)
                            # continue with the traverse
                            self._traverse(cmp, traversed_out_params)

    def traverse(self) -> None:
        """Traverse the components graph before execution."""
        # find start nodes
        self._seq = []
        traversed_out_params: Set[Param] = set()

        for component in self.nodes:
            # look for the component that has no input pins (only components with connections will be aded)
            if len(component.inputs) == 0 and len(component.outputs.get_params(True)):
                self._seq.append(component)
                self._traverse(component, traversed_out_params)

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
            for obj in self._seq:
                # check data types for the input params (probably this check can be removed)
                for p in obj.inputs:
                    typeguard.check_type(f"{obj.name}.{p.name}", p.type, p.value)
                # exec the component
                state = obj()

                if state == ComponentState.STOPPED:
                    log.info(f"Component {obj.name} stopped the pipeline.")
                    break
                if state == ComponentState.DISABLED:
                    log.warning(f"Component {obj.name} is DISABLED.")
                if state == ComponentState.ERROR:
                    log.error(f"Component {obj.name} produced an ERROR.")
                    break

            # code to run before running the next iteration in the pipeline
            for obj in self._seq:
                obj.finish_iter()
            if state == ComponentState.STOPPED:
                log.info("DONE")
                break
            time.sleep(2)
