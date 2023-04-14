"""Component definition."""
from __future__ import annotations
from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Callable, Type, Any, Coroutine
import logging
import asyncio
import traceback
import functools

try:
    import torch.nn as nn
except ImportError:
    pass

from limbus_config import config
from limbus.core.params import Params, InputParams, OutputParams
from limbus.core.states import ComponentState, ComponentStoppedError
# Note that Pipeline class cannot be imported to avoid circular dependencies.
if TYPE_CHECKING:
    from limbus.core.pipeline import Pipeline

log = logging.getLogger(__name__)


base_class: Type = object
if config.COMPONENT_TYPE == "generic":
    pass
elif config.COMPONENT_TYPE == "torch":
    try:
        base_class = nn.Module
    except NameError:
        log.error("Torch not installed. Using generic base class.")
else:
    log.error("Invalid component type. Using generic base class.")


# this is a decorator that will determine how many iterations must be run
def executions_manager(func: Callable) -> Callable:
    """Update the last execution to be run by the component."""
    @functools.wraps(func)
    async def wrapper_set_iteration(self, *args, **kwargs):
        if self.pipeline is not None:
            self.stopping_execution = self.pipeline.get_component_stopping_iteration(self)
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
        self._message: None | str = None
        self._component: Component = component
        self._verbose: bool = verbose

    def __call__(self, state: None | ComponentState = None, msg: None | str = None) -> ComponentState:
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
            self._message = msg
            self._logger()
        return self._state

    def _logger(self) -> None:
        """Log the message with the component name, iteration number and state."""
        if self._verbose:
            if self._message is None:
                log.info(f" {self._component.name}({self._component.executions_counter}): {self._state.name}")
            else:
                log.info(f" {self._component.name}({self._component.executions_counter}): {self._state.name}"
                         f" ({self._message})")

    @property
    def message(self) -> None | str:
        """Get the message associated to the state."""
        return self._message

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


class Component(base_class):
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
        self.__state: _ComponentState = _ComponentState(self, ComponentState.INITIALIZED)
        self.__pipeline: None | Pipeline = None
        self.__exec_counter: int = 0  # Counter of executions.
        # Last execution to be run in the __call__ loop.
        self.__stopping_execution: int = 0  # 0 means run forever

        # method called in _run_with_hooks to execute the component forward method
        self.__run_forward: Callable[..., Coroutine[Any, Any, ComponentState]] = self.forward
        try:
            if nn.Module in Component.__mro__:
                # If the component inherits from nn.Module, the forward method is called by the __call__ method
                self.__run_forward = partial(nn.Module.__call__, self)
        except NameError:
            pass

    def init_from_component(self, ref_component: Component) -> None:
        """Init basic execution params from another component.

        Args:
            ref_component: reference component.

        """
        self.__pipeline = ref_component.__pipeline
        if self.__pipeline is not None:
            self.__pipeline.add_nodes(self)
        self.verbose = ref_component.verbose

    @property
    def executions_counter(self) -> int:
        """Get the executions counter."""
        return self.__exec_counter

    @property
    def stopping_execution(self) -> int:
        """Get the last execution to be run by the component in the __call__ loop.

        Note that extra executions can be forced by other components to be able to run their executions.

        """
        return self.__stopping_execution

    @stopping_execution.setter
    def stopping_execution(self, value: int) -> None:
        """Set the last execution to be run by the component in the __call__ loop.

        Note that extra executions can be forced by other components to be able to run their executions.

        """
        self.__stopping_execution = value

    @property
    def state(self) -> tuple[ComponentState, None | str]:
        """Get the current state of the component and its associated message."""
        return (self.__state.state, self.__state.message)

    def set_state(self, state: ComponentState, msg: None | str = None) -> None:
        """Set the state of the component.

        Args:
            state: state to set.
            msg (optional): message to log.

        """
        self.__state(state, msg)

    @property
    def verbose(self) -> bool:
        """Get the verbose state."""
        return self.__state.verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set the verbose state."""
        self.__state.verbose = value

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
        properties: list[str] = self._properties.get_params()
        for key, value in kwargs.items():
            if key in properties:
                self._properties.set_param(key, value)
            else:
                log.warning(f"In component {self._name} the param {key} is not a valid viz param.")
                all_ok = False
        return all_ok

    @property
    def pipeline(self) -> None | Pipeline:
        """Get the pipeline object."""
        return self.__pipeline

    def set_pipeline(self, pipeline: None | Pipeline) -> None:
        """Set the pipeline running the component."""
        self.__pipeline = pipeline

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

    @executions_manager
    async def __call__(self) -> None:
        """Execute the forward method.

        If the component is executed in a pipeline, the component runs forever. However,
        if the component is run alone it will run only once.

        NOTE 1: If you want to use `async for...` instead of `while True` this method must be overridden.
        E.g.:
            async for x in xyz:
                if await self._run_with_hooks(x):
                    break

            Note that in this example the forward method will require 1 parameter.

        NOTE 2: if you override this method you must add the `executions_manager` decorator.

        """
        while True:
            if await self._run_with_hooks():
                break

    def is_stopped(self) -> bool:
        """Check if the component is stopped or is going to be stopped."""
        if self.state[0] in [ComponentState.STOPPED, ComponentState.STOPPED_AT_ITER,
                             ComponentState.ERROR, ComponentState.FORCED_STOP,
                             ComponentState.STOPPED_BY_COMPONENT]:
            return True
        return False

    def _stop_if_needed(self) -> bool:
        """Stop the component if it is required."""
        if self.is_stopped():
            if self.state[0] is not ComponentState.STOPPED_AT_ITER:
                # in this case we need to force the stop of the component. When it is stopped at a given iter
                # the pipeline ends without forcing anything.
                self._stop_component()
            return True
        return False

    async def _run_with_hooks(self, *args, **kwargs) -> bool:
        self.__exec_counter += 1
        if self.__pipeline is not None:
            await self.__pipeline.before_component_hook(self)
            if self._stop_if_needed():
                return True
        # run the component
        try:
            if len(self._inputs) == 0:
                # RUNNING state is set once the input params are received, if there are not inputs the state is set here
                if self.__pipeline is not None and self.__pipeline.before_component_user_hook:
                    await self.__pipeline.before_component_user_hook(self)
                self.set_state(ComponentState.RUNNING)
            self.set_state(await self.__run_forward(*args, **kwargs))
        except ComponentStoppedError as e:
            self.set_state(e.state)
        except Exception as e:
            self.set_state(ComponentState.ERROR, f"{type(e).__name__} - {str(e)}")
            log.error(f"Error in component {self.name}.\n"
                      f"{''.join(traceback.format_exception(None, e, e.__traceback__))}")
        if self.__pipeline is not None:
            # after component hook
            await self.__pipeline.after_component_hook(self)
            if self.__pipeline.after_component_user_hook:
                await self.__pipeline.after_component_user_hook(self)
            if self._stop_if_needed():
                return True
            return False
        # if there is not a pipeline, the component is executed only once
        return True

    @abstractmethod
    async def forward(self, *args, **kwargs) -> ComponentState:
        """Run the component, this method shouldn't be called, instead call __call__."""
        raise NotImplementedError
