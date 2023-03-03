"""Component definition."""
from __future__ import annotations
from abc import abstractmethod
from typing import List, Optional, TYPE_CHECKING, Callable, Type
import logging
import asyncio
import traceback
import functools

try:
    import torch.nn as nn
    base_cls: Type = nn.Module
except ImportError:
    base_cls = object

from limbus.core.params import Params, InputParams, OutputParams
from limbus.core.states import ComponentState, ComponentStoppedError
# Note that Pipeline class cannot be imported to avoid circular dependencies.
if TYPE_CHECKING:
    from limbus.core.pipeline import Pipeline

log = logging.getLogger(__name__)


class Module(base_cls):  # noqa: D101
    pass


# this is a decorator that will determine how many iterations must be run
def iterations_manager(func: Callable) -> Callable:
    """Update the last iteration to be run by the component."""
    @functools.wraps(func)
    async def wrapper_set_iteration(self, *args, **kwargs):
        if self._pipeline is not None:
            self._stopping_iteration = self._pipeline.get_component_stopping_iteration(self)
        return await func(self, *args, **kwargs)
    return wrapper_set_iteration


class _ComponentState():
    """Manage the state of the component.

    Args:
        component (Component): component to manage.
        state (ComponentState): initial state.
        verbose (bool, optional): verbose state. Default: False.

    """
    def __init__(self, component: Component, state: ComponentState, verbose: bool = False):
        self._state: ComponentState = state
        self._component: Component = component
        self._verbose: bool = verbose

    def __call__(self, state: Optional[ComponentState] = None, msg: Optional[str] = None) -> ComponentState:
        """Set the state of the component.

        If no args are passed, it returns the current state.

        Args:
            state (optional): state to set. Default: None.
            msg (optional): message to log. Default: None.

        Returns:
            The state of the component.

        """
        if state is not None:
            self._state = state
            self._logger(self._component.name, self._component.counter, self._state, msg)
        return self._state

    def _logger(self, comp_name: str, iters: int, state: ComponentState, msg: Optional[str]) -> None:
        """Log the message with the component name, iteration number and state."""
        if self._verbose:
            if msg is None:
                log.info(f" {comp_name}({iters}): {state.name}")
            else:
                log.info(f" {comp_name}({iters}): {state.name} ({msg})")

    @property
    def state(self) -> ComponentState:
        """Get the state of the component."""
        return self._state

    @property
    def verbose(self) -> bool:
        """Get the verbose state."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbose state."""
        self._verbose = value


class Component(Module):
    """Base class to define a Limbus Component.

    Args:
        name (str): component name.

    """

    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self._inputs = InputParams(self)
        self.__class__.register_inputs(self._inputs)
        self._outputs = OutputParams(self)
        self.__class__.register_outputs(self._outputs)
        self._properties = Params(self)
        self.__class__.register_properties(self._properties)
        self._resume_event: Optional[asyncio.Event] = None
        self._state: _ComponentState = _ComponentState(self, ComponentState.INITIALIZED)
        self._pipeline: Optional[Pipeline] = None
        self._exec_counter: int = 0  # Counter of executions.
        # Last execution to be run in the __call__ loop.
        self._stopping_iteration: int = 0  # 0 means run forever

    def init_from_component(self, ref_component: Component) -> None:
        """Init basic execution params from another component.

        Args:
            ref_component: reference component.

        """
        self._pipeline = ref_component._pipeline
        if self._pipeline is not None:
            self._pipeline.add_nodes(self)
        self.verbose = ref_component.verbose

    @property
    def counter(self) -> int:
        """Get the executions counter."""
        return self._exec_counter

    @property
    def stopping_iteration(self) -> int:
        """Get the last iteration to be run by the component in the __call__ loop."""
        return self._stopping_iteration

    @property
    def state(self) -> ComponentState:
        """Get the current state of the component."""
        return self._state.state

    def set_state(self, state: ComponentState, msg: Optional[str] = None) -> None:
        """Set the state of the component.

        Args:
            state: state to set.
            msg (optional): message to log.

        """
        self._state(state, msg)

    @property
    def verbose(self) -> bool:
        """Get the verbose state."""
        return self._state.verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbose state."""
        self._state.verbose = value

    @property
    def name(self) -> str:
        """Name of the component."""
        return self._name

    @property
    def inputs(self) -> InputParams:
        """Get the set of component inputs."""
        return self._inputs

    @property
    def outputs(self) -> OutputParams:
        """Get the set of component outputs."""
        return self._outputs

    @property
    def properties(self) -> Params:
        """Get the set of properties for this component."""
        return self._properties

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the input params.

        Args:
            inputs: Params object to register the inputs.

        """
        pass

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the output params.

        Args:
            outputs: Params object to register the outputs.

        """
        pass

    @staticmethod
    def register_properties(properties: Params) -> None:
        """Register the properties.

        These params are optional.

        Args:
            properties: Params object to register the properties.

        """
        pass

    def set_properties(self, **kwargs) -> bool:
        """Simplify the way to set the viz params.

        You can pass all the viz params you want to set as keyword arguments.

        These 2 codes are equivalent:
        >> component.set_properties(param_name_0=value_0, param_name_1=value_1, ...)

        and
        >> component.properties.set_param('param_name_0', value_0)
        >> component.properties.set_param('param_name_1', value_1)
        >> .
        >> .

        Returns:
            bool: True if all the passed viz params were setted, False otherwise.

        """
        all_ok = True
        properties: List[str] = self._properties.get_params()
        for key, value in kwargs.items():
            if key in properties:
                self._properties.set_param(key, value)
            else:
                log.warning(f"In component {self._name} the param {key} is not a valid viz param.")
                all_ok = False
        return all_ok

    @property
    def pipeline(self) -> Optional[Pipeline]:
        """Get the pipeline object."""
        return self._pipeline

    def set_pipeline(self, pipeline: Optional[Pipeline]) -> None:
        """Set the pipeline running the component."""
        self._pipeline = pipeline

    def _stop_component(self) -> None:
        """Prepare the component to be stopped."""
        for input in self._inputs.get_params():
            for ref in self._inputs[input].references:
                assert ref.sent is not None
                assert ref.consumed is not None
                # unblock the events
                ref.sent.set()
                ref.consumed.set()
        for output in self._outputs.get_params():
            for ref in self._outputs[output].references:
                assert ref.sent is not None
                assert ref.consumed is not None
                # unblock the events
                ref.sent.set()
                ref.consumed.set()

    @iterations_manager
    async def __call__(self) -> None:
        """Execute the forward method.

        If the component is executed in a pipeline, the component runs forever. However,
        if the component is run alone it will run only once.

        NOTE: If you want to use `async for...` instead of `while True` this method must be overridden.
        E.g.:
            async for x in xyz:
                if await self._run_with_hooks(x):
                    break

            Note that in this example the forward method will require 1 parameter.

        """
        while True:
            if await self._run_with_hooks():
                break

    def is_stopped(self) -> bool:
        """Check if the component is stopped or is going to be stopped."""
        if self.state in [ComponentState.STOPPED, ComponentState.STOPPED_AT_ITER,
                          ComponentState.ERROR, ComponentState.FORCED_STOP]:
            return True
        return False

    def _stop_if_needed(self) -> bool:
        """Stop the component if it is required."""
        if self.is_stopped():
            if self.state is not ComponentState.STOPPED_AT_ITER:
                # in this case we need to force the stop of the component. When it is stopped at a given iter
                # the pipeline ends without forcing anything.
                self._stop_component()
            return True
        return False

    async def _run_with_hooks(self, *args, **kwargs) -> bool:
        self._exec_counter += 1
        if self._pipeline is not None:
            await self._pipeline.before_component_hook(self)
            if self._stop_if_needed():
                return True
        # run the component
        try:
            if len(self._inputs) == 0:
                # RUNNING state is set once the input params are received, if there are not inputs the state is set here
                self.set_state(ComponentState.RUNNING)
            if hasattr(super(), "__call__"):
                # If the component inherits from nn.Module, the forward method is called by the __call__ method
                self.set_state(await super().__call__(*args, **kwargs))
            else:
                self.set_state(await self.forward(*args, **kwargs))
        except ComponentStoppedError as e:
            self.set_state(e.state)
        except Exception as e:
            self.set_state(ComponentState.ERROR, f"{type(e).__name__} - {str(e)}")
            log.error(f"Error in component {self.name}.\n"
                      f"{''.join(traceback.format_exception(None, e, e.__traceback__))}")
        if self._pipeline is not None:
            # after component hook
            await self._pipeline.after_component_hook(self)
            if self._stop_if_needed():
                return True
            return False
        # if there is not a pipeline, the component is executed only once
        return True

    @abstractmethod
    async def forward(self, *args, **kwargs) -> ComponentState:
        """Run the component, this method shouldn't be called, instead call __call__."""
        raise NotImplementedError

    def finish_iter(self) -> None:
        """Event executed when each pipeline iter finishes.

        Note that this method can be defined as async if needed.

        """
        pass

    def finish_pipeline(self) -> None:
        """Event executed when the pipeline finishes.

        Note that this method can be defined as async if needed.

        """
        pass

    def init_iter(self) -> None:
        """Event executed when each pipeline iter starts.

        Note that this method can be defined as async if needed.

        """
        pass

    def init_pipeline(self) -> None:
        """Event executed when the pipeline starts.

        Note that this method can be defined as async if needed.
        For example this method is useful to init some await variables that cannot be initialized in the constructor.

        """
        pass
