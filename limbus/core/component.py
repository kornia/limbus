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
from limbus.core.params import (InputParams, OutputParams, PropertyParams, InputEvents, OutputEvents, InputEvent,
                                EventType)
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


# Component EventTypes
class BeforeComponentCallEventType(EventType):
    """Denote a special type of event waited before the execution of the Component."""
    pass

class BeforeComponentIterEventType(EventType):
    """Denote a special type of event waited before each Component iteration."""
    pass

class AfterComponentIterEventType(EventType):
    """Denote a special type of event waited after each Component iteration."""
    pass


class _ComponentState():
    """Manage the state of the component.

    Note that the state can be multiple.
    The user interactions are the ones allowing simultaneous states, concretelly:
        - ComponentState.STOPPED_AT_ITER
        - ComponentState.STOPPED_BY_COMPONENT (it is generated by the STOPPED_AT_ITER state in other components)
        - ComponentState.FORCED_STOP
    E.g., a component can be properly executed and at the same time stopped by the user.

    Args:
        component (Component): component to manage.
        state (ComponentState): initial state.
        verbose (bool, optional): verbose state. Default: False.

    """
    def __init__(self, component: Component, state: ComponentState, verbose: bool = False):
        self._states: list[ComponentState] = [state]
        self._messages: dict[ComponentState, None | str] = {state: None}
        self._component: Component = component
        self._verbose: bool = verbose

    def __call__(self, state: None | ComponentState = None, msg: None | str = None, add: bool = False
                 ) -> list[ComponentState]:
        """Set or add a new state for the component.

        If no args are passed, it returns the current state.

        Args:
            state (optional): state to set. Default: None.
            msg (optional): message to log. Default: None.
            add (optional): if True, the state is added to the list of states. Default: False.

        Returns:
            The state or states of the component.

        """
        if state is not None:
            if add:
                self._states.append(state)
                self._messages[state] = msg
            else:
                self._states = [state]
                self._messages = {state: msg}
            self._logger()
        return self._states

    def _logger(self) -> None:
        """Log the message with the component name, iteration number and state."""
        if self._verbose:
            num_states = len(self._states)
            for idx, state in enumerate(self._states):
                msg = self._messages.get(state, None)
                msg_str: str = f""
                if num_states > 1:
                    msg_str = f" {idx + 1}/{num_states}"
                msg_str = f"{msg_str} {self._component.name}({self._component.executions_counter}): {state.name}"
                if msg is not None:
                    # concat the message
                    msg_str = f"{msg_str} ({msg})"
                log.info(msg_str)

    def message(self, state: ComponentState) -> None | str:
        """Get the message associated to the state. If state is not found, returns None."""
        return self._messages.get(state, None)

    @property
    def state(self) -> list[ComponentState]:
        """Get the state of the component."""
        return self._states

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
        self._properties = PropertyParams(self)
        self.__class__.register_properties(self._properties)
        self._input_events = InputEvents(self)
        self.__class__.register_input_events(self._input_events)
        self._output_events = OutputEvents(self)
        self.__class__.register_output_events(self._output_events)
        self.__state: _ComponentState = _ComponentState(self, ComponentState.INITIALIZED)
        self.__pipeline: None | Pipeline = None
        self.__exec_counter: int = 0  # Counter of executions.
        # Last execution to be run in the __call__ loop.
        self.__stopping_execution: int = 0  # 0 means run forever
        self.__num_params_waiting_to_receive: int = 0  # updated from InputParam

        # by default Components can wait for events at 3 execution points:
        # - before running the component in the __call__ method
        self.__events_to_wait_before_call: InputEvents =  InputEvents(self)
        # - before running the component in the __run_with_hooks method
        self.__events_to_wait_before_running: InputEvents = InputEvents(self)
        # - after running the component in the __run_with_hooks method
        self.__events_to_wait_after_running: InputEvents =  InputEvents(self)
        # NOTE that other event types can be managed by the user. Defining the event type as EventType.
        # assign the events to the corresponding event types
        for event in self._input_events:
            if isinstance(event, BeforeComponentCallEventType):
                self.__events_to_wait_before_call.__setattr__(event.name, event)
            elif isinstance(event, BeforeComponentIterEventType):
                self.__events_to_wait_before_running.__setattr__(event.name, event)
            elif isinstance(event, AfterComponentIterEventType):
                self.__events_to_wait_after_running.__setattr__(event.name, event)

        # method called in __run_with_hooks to execute the component forward method
        self.__run_forward: Callable[..., Coroutine[Any, Any, ComponentState]] = self.forward
        try:
            if nn.Module in Component.__mro__:
                # If the component inherits from nn.Module, the forward method is called by the __call__ method
                self.__run_forward = partial(nn.Module.__call__, self)
        except NameError:
            pass

    def __del__(self):
        self.release()

    def release(self) -> None:
        """Event executed when the component ends its execution."""
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
    def state(self) -> list[ComponentState]:
        """Get the current state/s of the component."""
        return self.__state.state

    def state_message(self, state: ComponentState) -> None | str:
        """Get the message associated a given current state of the component."""
        return self.__state.message(state)

    def set_state(self, state: ComponentState, msg: None | str = None, add: bool = False) -> None:
        """Set the state of the component.

        Args:
            state: state to set.
            msg (optional): message to log. Default: None.
            add (optional): if True, the state is added to the list of states. Default: False.

        """
        self.__state(state, msg, add)

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
    def properties(self) -> PropertyParams:
        """Get the set of properties for this component."""
        return self._properties

    @property
    def input_events(self) -> InputEvents:
        """Get the set of input events for this component."""
        return self._input_events

    @property
    def output_events(self) -> OutputEvents:
        """Get the set of output events for this component."""
        return self._output_events

    @staticmethod
    def register_inputs(inputs: InputParams) -> None:
        """Register the input params.

        Args:
            inputs: object to register the inputs.

        """
        pass

    @staticmethod
    def register_outputs(outputs: OutputParams) -> None:
        """Register the output params.

        Args:
            outputs: object to register the outputs.

        """
        pass

    @staticmethod
    def register_properties(properties: PropertyParams) -> None:
        """Register the properties.

        These params are optional.

        Args:
            properties: object to register the properties.

        """
        pass

    @staticmethod
    def register_input_events(inputs: InputEvents) -> None:
        """Register the input events.

        Args:
            inputs: object to register the input events.

        """
        pass

    @staticmethod
    def register_output_events(outputs: OutputEvents) -> None:
        """Register the output events.

        Args:
            outputs: object to register the output events.

        """
        pass

    @property
    def pipeline(self) -> None | Pipeline:
        """Get the pipeline object."""
        return self.__pipeline

    def set_pipeline(self, pipeline: None | Pipeline) -> None:
        """Set the pipeline running the component."""
        self.__pipeline = pipeline

    def __stop_component(self) -> None:
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
                if await self.__run_with_hooks(x):
                    break

            Note that in this example the forward method will require 1 parameter.

        NOTE 2: if you override this method you must add the `executions_manager` decorator.

        """
        if await self.__gather_events(self.__events_to_wait_before_call) is True:
            return
        while True:
            if await self.__run_with_hooks():
                break

    def is_stopped(self) -> bool:
        """Check if the component is stopped or is going to be stopped."""
        if len(set(self.state).intersection(set([ComponentState.STOPPED, ComponentState.STOPPED_AT_ITER,
                                                 ComponentState.ERROR, ComponentState.FORCED_STOP,
                                                 ComponentState.STOPPED_BY_COMPONENT]))) > 0:
            return True
        return False

    def __stop_if_needed(self) -> bool:
        """Stop the component if it is required."""
        if self.is_stopped():
            if ComponentState.STOPPED_AT_ITER not in self.state:
                # in this case we need to force the stop of the component. When it is stopped at a given iter
                # the pipeline ends without forcing anything.
                self.__stop_component()
            return True
        return False

    async def __gather_events(self, events: InputEvents) -> bool | None:
        """Return true/false wether execution must finish now or not and None if it is unknown."""
        if len(events) > 0:
            try:
                states = await asyncio.gather(*[event.wait() for event in events if isinstance(event, InputEvent)])
                # if the event is not connected to any other event
                if states.count(None) == len(states):
                     return None
                # if the event is connected to disabled events
                elif states.count(False) == len(states):
                    return True
                return False
            except ComponentStoppedError as e:
                self.set_state(e.state, e.message, add=True)
                return True
        return None

    async def __run_with_hooks(self, *args, **kwargs) -> bool:
        if await self.__gather_events(self.__events_to_wait_before_running) is True:
            return True
        self.__exec_counter += 1
        if self.__pipeline is not None:
            await self.__pipeline.before_component_hook(self)
            if self.__pipeline.before_component_user_hook:
                await self.__pipeline.before_component_user_hook(self)
            if self.__stop_if_needed():  # just in case the component state is changed in the before_component_hook
                return True
        # run the component
        try:
            if len(self._inputs) == 0:
                # RUNNING state is set once the input params are received, if there are not inputs the state is set here
                self.set_state(ComponentState.RUNNING)
            self.set_state(await self.__run_forward(*args, **kwargs))
        except ComponentStoppedError as e:
            self.set_state(e.state, e.message, add=True)
        except Exception as e:
            self.set_state(ComponentState.ERROR, f"{type(e).__name__} - {str(e)}")
            log.error(f"Error in component {self.name}.\n"
                      f"{''.join(traceback.format_exception(None, e, e.__traceback__))}")
        ret_val = True
        if self.__pipeline is not None:
            # after component hook
            await self.__pipeline.after_component_hook(self)
            if self.__pipeline.after_component_user_hook:
                await self.__pipeline.after_component_user_hook(self)
            if not self.__stop_if_needed():
                ret_val = False
        # NOTE that evetns without pipeline are not properly controlled
        res = await self.__gather_events(self.__events_to_wait_after_running)
        if res is not None:
            ret_val = res
        # if there is not a pipeline, the component is executed only once
        return ret_val if self.__pipeline is not None else True

    @abstractmethod
    async def forward(self, *args, **kwargs) -> ComponentState:
        """Run the component, this method shouldn't be called, instead call __call__."""
        raise NotImplementedError
