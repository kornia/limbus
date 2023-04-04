"""Components manager to connect, traverse and execute pipelines."""
from __future__ import annotations
from typing import Coroutine, Any, Callable
import logging
import asyncio

from limbus.core.component import Component, ComponentState
from limbus.core.states import PipelineState, VerboseMode
from limbus.core import async_utils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class _PipelineState():
    """Manage the state of the pipeline."""
    def __init__(self, state: PipelineState, verbose: VerboseMode = VerboseMode.DISABLED):
        self._state: PipelineState = state
        self._verbose: VerboseMode = verbose

    def __call__(self, state: PipelineState, msg: None | str = None) -> None:
        """Set the state of the pipeline.

        Args:
            state: state to set.
            msg (optional): message to log. Default: None.

        """
        self._state = state
        self._logger(self._state, msg)

    def _logger(self, state: PipelineState, msg: None | str) -> None:
        """Log the message with the pipeline state."""
        if self._verbose != VerboseMode.DISABLED:
            if msg is None:
                log.info(f" {state.name}")
            else:
                log.info(f" {state.name}: {msg}")

    @property
    def state(self) -> PipelineState:
        """Get the state of the pipeline."""
        return self._state

    @property
    def verbose(self) -> VerboseMode:
        """Get the verbose mode."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: VerboseMode) -> None:
        """Set the verbose mode."""
        self._verbose = value


class Pipeline:
    """Class to create and execute a pipeline of Limbus Components."""
    def __init__(self) -> None:
        self._nodes: set[Component] = set()  # note that it does not contain all the nodes
        self._resume_event: asyncio.Event = asyncio.Event()
        self._stop_event: asyncio.Event = asyncio.Event()
        self._state: _PipelineState = _PipelineState(PipelineState.INITIALIZING)
        self._counter: int = 0  # number of iterations executed in the pipeline (== component with more executions).
        # Number of times each component will be run at least.
        # This feature should be mainly used for debugging purposes. It can make the processing a bit slower and
        # depending on the graph to be executed it can require to recreate tasks (e.g. when a given component requires
        # several runs from a previous one).
        self._min_number_of_iters_to_run: int = 0  # 0 means until the end of the pipeline
        # user defined hooks
        self._before_component_user_hook: None | Callable = None
        self._after_component_user_hook: None | Callable = None
        self._before_iteration_user_hook: None | Callable = None
        self._after_iteration_user_hook: None | Callable = None
        self._before_pipeline_user_hook: None | Callable = None
        self._after_pipeline_user_hook: None | Callable = None

    def set_before_pipeline_user_hook(self, hook: None | Callable) -> None:
        """Set a hook to be executed before the pipeline execution.

        This callable must have a single parameter which is the state of the pipeline at the begining of the pipeline.
        Moreover it must be async.

        Prototype: async def hook_name(state: PipelineState).
        """
        self._before_pipeline_user_hook = hook

    @property
    def before_pipeline_user_hook(self) -> None | Callable:
        """Get the before pipeline user hook."""
        return self._before_pipeline_user_hook

    def set_after_pipeline_user_hook(self, hook: None | Callable) -> None:
        """Set a hook to be executed after the pipeline execution.

        This callable must have a single parameter which is the state of the pipeline at the end of the pipeline.
        Moreover it must be async.

        Prototype: async def hook_name(state: PipelineState).
        """
        self._after_pipeline_user_hook = hook

    @property
    def after_pipeline_user_hook(self) -> None | Callable:
        """Get the after pipeline user hook."""
        return self._after_pipeline_user_hook

    def set_before_iteration_user_hook(self, hook: None | Callable) -> None:
        """Set a hook to be executed before each iteration.

        This callable must have a single parameter which is an int denoting the iter being executed.
        Moreover it must be async.

        Prototype: async def hook_name(counter: int).
        """
        self._before_iteration_user_hook = hook

    @property
    def before_iteration_user_hook(self) -> None | Callable:
        """Get the before iteration user hook."""
        return self._before_iteration_user_hook

    def set_after_iteration_user_hook(self, hook: None | Callable) -> None:
        """Set a hook to be executed after each iteration.

        This callable must have a single parameter which is the state of the pipeline at the end of the iteration.
        Moreover it must be async.

        Prototype: async def hook_name(state: PipelineState).
        """
        self._after_iteration_user_hook = hook

    @property
    def after_iteration_user_hook(self) -> None | Callable:
        """Get the after iteration user hook."""
        return self._after_iteration_user_hook

    def set_before_component_user_hook(self, hook: None | Callable) -> None:
        """Set a hook to be executed before each component.

        This callable must have a single parameter which is the component being executed.
        Moreover it must be async.

        Prototype: async def hook_name(obj: Componet).
        """
        self._before_component_user_hook = hook

    @property
    def before_component_user_hook(self) -> None | Callable:
        """Get the before component user hook."""
        return self._before_component_user_hook

    def set_after_component_user_hook(self, hook: None | Callable) -> None:
        """Set a hook to be executed after each component.

        This callable must have a single parameter which is the component being executed.
        Moreover it must be async.

        Prototype: async def hook_name(obj: Componet).
        """
        self._after_component_user_hook = hook

    @property
    def after_component_user_hook(self) -> None | Callable:
        """Get the after component user hook."""
        return self._after_component_user_hook

    def get_component_stopping_iteration(self, component: Component) -> int:
        """Compute the iteration where the __call__ loop of the component will be stopped.

        Args:
            component: component to be run.

        Returns:
            int denoting the iteration where the _call__ loop will be stopped.
            0 means that it will run forever.

        """
        if self._min_number_of_iters_to_run > 0:
            return component.counter + self._min_number_of_iters_to_run
        return 0

    async def before_component_hook(self, component: Component) -> None:
        """Run before the execution of each component.

        Args:
            component: component to be executed.

        """
        if not self._resume_event.is_set():
            component.set_state(ComponentState.PAUSED)
        await self._resume_event.wait()
        component.set_state(ComponentState.READY)

    async def after_component_hook(self, component: Component) -> None:
        """Run after the execution of each component.

        Args:
            component: executed component.

        """
        # determine when the component must be stopped
        # when the pipeline claims that it must be stopped...
        if self._stop_event.is_set():
            component.set_state(ComponentState.FORCED_STOP)
            return
        # when the number of iters to run is reached...
        # since each component can be running a different iteration we assign the max value
        self._counter = max(self.counter, component.counter)
        if self._min_number_of_iters_to_run != 0 and component.counter >= component.stopping_iteration:
            component.set_state(ComponentState.STOPPED_AT_ITER)

    @property
    def counter(self) -> int:
        """Get the number of started pipeline iterations."""
        return self._counter

    @property
    def state(self) -> PipelineState:
        """Get the state of the pipeline."""
        return self._state.state

    def add_nodes(self, components: Component | list[Component]) -> None:
        """Add components to the pipeline.

        Note: At least one component per graph must be added to be able to run the pipeline. The pipeline will
        automatically add the nodes that are missing at the begining.

        Args:
            components: Component or list of components to be added.

        """
        if isinstance(components, Component):
            components = [components]
        for component in components:
            self._nodes.add(component)

    def pause(self) -> None:
        """Pause the execution of the pipeline.

        Note: Components will be paused as soon as posible, if the pipeline is running will be done inmediatelly after
        sending the outputs. Some components waiting for inputs will remain in that state since the previous components
        can be paused.
        """
        if self._resume_event.is_set():
            self._state(PipelineState.PAUSED)
            self._resume_event.clear()

    def stop(self) -> None:
        """Force the stop of the pipeline."""
        self.resume()  # if the pipeline is paused it is blocked
        self._stop_event.set()  # stop the forever loop inside each component
        self._state(PipelineState.FORCED_STOP)

    def resume(self) -> None:
        """Resume the execution of the pipeline."""
        if not self._resume_event.is_set():
            self._state(PipelineState.RUNNING)
            self._resume_event.set()

    def set_verbose_mode(self, state: VerboseMode) -> None:
        """Set the verbose mode.

        Args:
            state: verbose mode to be set.

        """
        if self._state.verbose == state:
            return
        self._state.verbose = state
        for node in self._nodes:
            node.verbose = self._state.verbose == VerboseMode.COMPONENT

    async def async_run(self, iters: int = 0) -> PipelineState:
        """Run the components graph.

        Args:
            iters (optional): number of iters to be run. By default (0) all of them are run.

        Returns:
            PipelineState with the current pipeline status.

        """
        # Number of times each component will be run at least.
        # This feature should be mainly used for debugging purposes. It can make the processing a bit slower and
        # depending on the graph to be executed it can require to recreate tasks (e.g. when a given component requires
        # several runs from a previous one).
        self._min_number_of_iters_to_run = iters
        self._stop_event.clear()

        async def start() -> None:
            tasks: list[Coroutine[Any, Any, None]] = []
            for node in self._nodes:
                node.set_pipeline(self)
                tasks.append(node())
            self.resume()
            await asyncio.gather(*tasks)
            # check if there are pending tasks
            pending_tasks: list = []
            for node in self._nodes:
                t = async_utils.get_task_if_exists(node)
                if t is not None:
                    pending_tasks.append(t)
            await asyncio.gather(*pending_tasks)

        self._state(PipelineState.STARTED)
        if len(self._nodes) == 0:
            self._state(PipelineState.EMPTY, "No components added to the pipeline")

        if self.before_pipeline_user_hook is not None:
            # even if the pipeline is empty we run the hook
            await self.before_pipeline_user_hook(self.state)

        if self._state.state == PipelineState.EMPTY:
            # if it is empty we do not run the pipeline
            return self._state.state

        await start()

        # set the end state if there was not set before
        if self._state.state not in [PipelineState.FORCED_STOP, PipelineState.ERROR, PipelineState.EMPTY]:
            self._state(PipelineState.ENDED)
        if self.after_pipeline_user_hook is not None:
            await self.after_pipeline_user_hook(self.state)
        return self._state.state

    def run(self, iters: int = 0) -> PipelineState:
        """Run the components graph.

        Args:
            iters (optional): number of iters to be run. By default (0) all of them are run.

        Returns:
            PipelineState with the current pipeline status.

        """
        async_utils.run_coroutine(self.async_run(iters))
        return self._state.state
