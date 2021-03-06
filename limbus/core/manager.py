"""Components manager to connect, traverse and execute pipelines."""
from typing import List, Optional, Union, Set, Tuple, Callable
import logging
import time
import asyncio
from enum import Enum

import typeguard
import torch.nn as nn

from limbus.core import ComponentState, Component, Params, Param


class PipelineState(Enum):
    """Possible states for the pipeline."""
    STARTED = 0
    ENDED = 1
    PAUSED = 2
    ERROR = 3
    EMPTY = 4
    RUNNING = 5
    INITIALIZING = 6


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# TODO: check why I added this DefaultParam class, now it is not used
class DefaultParam():
    """Trick to denote when to use a default parameter."""
    pass


class Pipeline(nn.Module):
    """Class to create and execute a pipeline of Limbus Components."""
    def __init__(self):
        self.nodes: Set[Component] = set()
        self._seq: List[Component] = []
        self._counter = 0  # iterations counter
        self._pause: bool = False

        self._before_component_hook: Optional[Callable] = None
        self._after_component_hook: Optional[Callable] = None
        self._before_iteration_hook: Optional[Callable] = None
        self._after_iteration_hook: Optional[Callable] = None
        self._before_pipeline_hook: Optional[Callable] = None
        self._after_pipeline_hook: Optional[Callable] = None

    def set_before_pipeline_hook(self, hook: Optional[Callable]) -> None:
        """Set a hook to be executed before the pipeline execution.

        This callable must have a single parameter which is the state of the pipeline at the begining of the pipeline.
        The callable must be async.

        Prototype: async def hook_name(state: PipelineState).
        """
        self._before_pipeline_hook = hook

    def set_after_pipeline_hook(self, hook: Optional[Callable]) -> None:
        """Set a hook to be executed after the pipeline execution.

        This callable must have a single parameter which is the state of the pipeline at the end of the pipeline.
        The callable must be async.

        Prototype: async def hook_name(state: PipelineState).
        """
        self._after_pipeline_hook = hook

    def set_before_iteration_hook(self, hook: Optional[Callable]) -> None:
        """Set a hook to be executed before each iteration.

        This callable must have a single parameter which is an int denoting the iter being executed.
        The callable must be async.

        Prototype: async def hook_name(counter: int).
        """
        self._before_iteration_hook = hook

    def set_after_iteration_hook(self, hook: Optional[Callable]) -> None:
        """Set a hook to be executed after each iteration.

        This callable must have a single parameter which is the state of the pipeline at the end of the iteration.
        The callable must be async.

        Prototype: async def hook_name(state: PipelineState).
        """
        self._after_iteration_hook = hook

    def set_before_component_hook(self, hook: Optional[Callable]) -> None:
        """Set a hook to be executed before each component.

        This callable must have a single parameter which is the component being executed.
        The callable must be async.

        Prototype: async def hook_name(obj: Componet).
        """
        self._before_component_hook = hook

    def set_after_component_hook(self, hook: Optional[Callable]) -> None:
        """Set a hook to be executed after each component.

        This callable must have 3 parameters which are the component being executed, its state
        after the execution and an optional param with an exception raised by the component.
        The callable must be async.

        Prototype: async def hook_name(obj: Componet, state: ComponentState, e: Optional[Exception] = None).
        """
        self._after_component_hook = hook

    def add_nodes(self, components: Union[Component, List[Component]]) -> None:
        """Add components to the pipeline.

        Args:
            components: Component or list of components to be added.

        """
        if isinstance(components, Component):
            components = [components]
        for component in components:
            self.nodes.add(component)

    def _all_connected(self, ori: Params, dst: Set[Param]) -> bool:
        for p in ori:
            # list of unconnected params. Useful when the input is subscriptable
            unconnected_params: List[Tuple["Param", Optional[int]]] = [ref for ref in p.references if ref[0] not in dst]
            # an input param can be only connected to one output param
            if p.ref_counter() == 0 or len(unconnected_params) > 0:
                return False
        return True

    def _traverse(self, component, traversed_out_params: Set[Param]) -> None:
        # add params of the component since it was already added to the sequence
        for out_param in component.outputs:
            traversed_out_params.add(out_param)

        # search for connected components
        for out_param in component.outputs:
            # search for the references of each output param across all the component
            in_param: Tuple["Param", Optional[int]]
            for in_param in out_param.references:
                for cmp in self.nodes:
                    if cmp not in self._seq:
                        # try to find a component connected with the current component and with all
                        # its inputs connected
                        if in_param[0] in cmp.inputs and self._all_connected(cmp.inputs, traversed_out_params):
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
            # look for and add the components without input pins
            if len(component.inputs) == 0:
                self._seq.append(component)
                self._traverse(component, traversed_out_params)

    def pause(self) -> None:
        """Pause the execution of the pipeline once the current iteration finishes."""
        self._pause = True

    async def async_execute(self, iters: Optional[int] = None) -> PipelineState:
        """Execute the components graph with hooks.

        Args:
            iters (optional): number of iters to be run. By default all of them are run.

        Returns:
            PipelineState with the current pipeline status.

        """
        pipe_state: PipelineState = PipelineState.STARTED
        if self._pause:
            pipe_state = PipelineState.PAUSED

        if len(self._seq) == 0:
            log.error("The pipeline is empty, it does not contain components to execute.")
            pipe_state = PipelineState.EMPTY

        if self._counter == 0 and self._before_pipeline_hook is not None:
            await self._before_pipeline_hook(pipe_state)

        # stop execution if the pipeline is empty
        if pipe_state == PipelineState.EMPTY:
            return pipe_state

        counter = self._counter - 1
        while not self._pause:
            if iters is not None and iters + counter < self._counter:
                break
            log.info(f"Iteration {self._counter}")
            if self._before_iteration_hook is not None:
                await self._before_iteration_hook(self._counter)
            self._counter += 1
            for obj in self._seq:
                # check data types for the input params (probably this check can be removed)
                for p in obj.inputs:
                    typeguard.check_type(f"{obj.name}.{p.name}", p.type, p.value)

                if self._before_component_hook is not None:
                    await self._before_component_hook(obj)

                # exec the component
                exception: Optional[Exception] = None
                try:
                    state = obj()
                except Exception as e:
                    exception = e
                    log.error(f"Error in component {obj.name}: {e}")
                    state = ComponentState.ERROR

                if self._after_component_hook is not None:
                    await self._after_component_hook(obj, state, exception)

                if state == ComponentState.STOPPED:
                    log.info(f"Component {obj.name} stopped the pipeline.")
                    break
                if state == ComponentState.DISABLED:
                    log.warning(f"Component {obj.name} is DISABLED.")
                if state == ComponentState.ERROR:
                    log.error(f"Component {obj.name} produced an ERROR.")
                    break
                if state == ComponentState.NotImplemented:
                    log.error(f"Component {obj.name} returned a NotImplemented state.")
                    break

            # code to run before running the next iteration in the pipeline
            for obj in self._seq:
                obj.finish_iter()

            # set the pipeline state after the iteration
            pipe_state = PipelineState.RUNNING
            if state == ComponentState.STOPPED:
                pipe_state = PipelineState.ENDED
                log.info("Pipeline finished.")
            elif state in [ComponentState.ERROR, ComponentState.NotImplemented]:
                pipe_state = PipelineState.ERROR
                log.info("Pipeline finished with an error.")

            if self._after_iteration_hook is not None:
                await self._after_iteration_hook(pipe_state)

            # finish the pipeline
            if pipe_state in [PipelineState.ENDED, PipelineState.ERROR]:
                if self._after_pipeline_hook is not None:
                    await self._after_pipeline_hook(pipe_state)
                break
        # disable the pause if was paused
        self._pause = False
        return pipe_state

    def execute(self, iters: Optional[int] = None) -> PipelineState:
        """Execute the components graph.

        Args:
            iters (optional): number of iters to be run. By default all of them are run.

        Returns:
            PipelineState with the current pipeline status.

        """
        return asyncio.run(self.async_execute(iters))
