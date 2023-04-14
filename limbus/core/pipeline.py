"""Components manager to connect, traverse and execute pipelines."""
from __future__ import annotations
from typing import Coroutine, Any, Callable
import logging
import asyncio

from limbus.core.component import Component, ComponentState
from limbus.core.states import PipelineState, VerboseMode, IterationState
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
        # flag x component denoting if it was run in the iteration
        self._iteration_component_state: dict[Component, tuple[IterationState, int]] = {}
        # number of iterations executed in the pipeline (== component with less executions).
        self._min_iteration_in_progress: int = 0
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
        self._pipeline_updates_from_component_lock = asyncio.Lock()

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
        """Set a hook to be executed after each iteration (next iter can be already in execution).

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
            return component.executions_counter + self._min_number_of_iters_to_run
        return 0

    def _get_iteration_status(self) -> tuple[bool, bool | None]:
        """Get the status of the iteration.

        Returns:
            bool: True if all the components have finished the current iteration.
            bool | None: True if the next iteration has started. None if it is undertemined.
                Next iter status cannot be determined until the previous iter is finished.

        """
        values = list(self._iteration_component_state.values())
        # get the state of the iteration
        prev_iteration_status = [
            # this cond is not always correct x component but it is correct for the pipeline.
            # state[1] can be > self._min_iteration_in_progress but remaining in the same iter however the min
            # exec_counter of all the components will be equal to the pipeline iter.
            # NOTE: this will not be true once we allow adding components to the pipeline during the execution.
            (state[0] == IterationState.COMPONENT_EXECUTED or state[1] > self._min_iteration_in_progress
             ) for state in values
        ]
        prev_status = sum(prev_iteration_status) == len(prev_iteration_status)
        if prev_status:
            next_iteration_status = [state[0] == IterationState.COMPONENT_IN_EXECUTION for state in values]
            return prev_status, sum(next_iteration_status) > 0
        return prev_status, None

    async def before_component_hook(self, component: Component) -> None:
        """Run before the execution of each component.

        Args:
            component: component to be executed.

        """
        # just in case in the future several components run in parallel (not now)
        await self._pipeline_updates_from_component_lock.acquire()
        try:
            if not self._resume_event.is_set():
                component.set_state(ComponentState.PAUSED)
            await self._resume_event.wait()
            component.set_state(ComponentState.READY)
            # state of the iteration
            is_prev_iter_finished, _ = self._get_iteration_status()

            # denote that this component is being executed in the current iteration
            self._iteration_component_state[component] = (IterationState.COMPONENT_IN_EXECUTION,
                                                          component.executions_counter)
            if self._min_iteration_in_progress == 0:
                # denote that the first iteration is starting
                self._min_iteration_in_progress = 1

            if is_prev_iter_finished:
                # previous iteration has finished but the current one started before or just now
                self._min_iteration_in_progress += 1
        finally:
            self._pipeline_updates_from_component_lock.release()

    async def after_component_hook(self, component: Component) -> None:
        """Run after the execution of each component.

        Args:
            component: executed component.

        """
        # just in case in the future several components run in parallel (not now)
        await self._pipeline_updates_from_component_lock.acquire()
        try:
            # determine when the component must be stopped
            # when the pipeline claims that it must be stopped...
            if self._stop_event.is_set():
                component.set_state(ComponentState.FORCED_STOP)
                return
            # denote that this component was already executed in the current iteration
            self._iteration_component_state[component] = (IterationState.COMPONENT_EXECUTED,
                                                          component.executions_counter)

            # NEXT CODE is disabled because we cannot know when an iteration starts.
            # get the state of the iteration
            # is_prev_iter_finished, _ = self._get_iteration_status()
            # if is_prev_iter_finished:
            #    if self._after_iteration_user_hook is not None:
            #        # Since the last component being executed changes its state in this method the
            #        # min iteration in progress is correct.
            #        await self._after_iteration_user_hook(self.state, self._min_iteration_in_progress)

            # when the number of iters to run is reached...
            # NOTE: component could be stopped before finishing the number of iterations since execution != iteration.
            # In that case the other components will force rerunning this one to run the required iterations.
            if self._min_number_of_iters_to_run != 0 and component.executions_counter >= component.stopping_execution:
                component.set_state(ComponentState.STOPPED_AT_ITER)
        finally:
            self._pipeline_updates_from_component_lock.release()

    @property
    def min_iteration_in_progress(self) -> int:
        """Get the number of the oldest iteration still being executed."""
        return self._min_iteration_in_progress

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
        self._iteration_component_state = {}
        self._stop_event.clear()

        async def start() -> None:
            tasks: list[Coroutine[Any, Any, None]] = []
            for node in self._nodes:
                node.set_pipeline(self)
                tasks.append(node())
                self._iteration_component_state[node] = (IterationState.COMPONENT_NOT_EXECUTED, 0)
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

        # NOTE about limitting the number of iterations and using iteration hooks.
        # In order to achieve both we need to block asyncio execution. The selected mechanism is using a loop in this
        # method. This means that it is not efficient and this feature should be mainly used for debugging purposes.
        # It can make the processing a bit slower and depending on the graph to be executed it can require to recreate
        # tasks (e.g. when a given component requires several runs to finish one iteration).
        # Even if you do not limit the number of iterations but you want to run the pipeline forever using
        # iteration hooks then the iterations must be run independently to be able to know when each iteration starts
        # and ends. NOTE that after_iteration_user_hook() could be run in after_component_hook() but without control on
        # when the next iteration starts, so we disabled.
        # ATTENTION: We recommend to use this feature only for debugging!!!
        self._min_number_of_iters_to_run = 0
        # if there are hooks then iters must be run one by one forever
        if self._before_iteration_user_hook is not None or self._after_iteration_user_hook is not None:
            self._min_number_of_iters_to_run = 1

        # If there are no hooks but there is a limit in the number of iters we only set 1 iters but running all the
        # required iters.
        if self._min_number_of_iters_to_run == 0:
            self._min_number_of_iters_to_run = iters
            iters = 1

        # run the pipeline as independent iterations. The loop is only run ince if there are no hooks.
        forever = iters == 0
        while forever or iters > 0:  # run until the pipeline is completed or there are no iters to run
            iters -= 1 if not forever else 0
            if self._before_iteration_user_hook is not None:
                await self._before_iteration_user_hook(self._min_iteration_in_progress)
            await start()
            if self._after_iteration_user_hook is not None:
                await self._after_iteration_user_hook(self.state)

            # set the end state if there was not set before
            if self._state.state not in [PipelineState.FORCED_STOP, PipelineState.ERROR, PipelineState.EMPTY]:
                self._state(PipelineState.ENDED)
                break

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
