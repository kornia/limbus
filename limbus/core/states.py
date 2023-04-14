"""Define the states for components/pipelines."""
from enum import Enum


class ComponentStoppedError(Exception):
    """Raised when trying to interact with a stopped component.

    Properties:
        state: state of the component when the error was raised.
        message: explanation of the error.

    """
    def __init__(self, state: "ComponentState"):
        self.state: ComponentState = state
        super().__init__()


class VerboseMode(Enum):
    """Possible states for the verbose in the pipeline objects."""
    DISABLED = 0
    PIPELINE = 1
    COMPONENT = 2


class ComponentState(Enum):
    """Possible states for the components."""
    STOPPED = 0  # when the stop is because of the component stops internally
    PAUSED = 1  # when the component is paused because the user requires it
    OK = 2  # when the iteration is executed normaly
    ERROR = 3  # when the stop is because of an error
    DISABLED = 4  # when the component is disabled for some reason (e.g. viz cannot be done)
    FORCED_STOP = 5  # when the stop is because the user requires it
    INITIALIZED = 6  # whe it is created
    RUNNING = 7
    RECEIVING_PARAMS = 8
    SENDING_PARAMS = 9
    STOPPED_AT_ITER = 10  # when the stop is because of the iteration number
    READY = 11  # when the component is ready to be executed at the beginning of each iteration
    STOPPED_BY_COMPONENT = 12  # when the stop is because another component forces it


class PipelineState(Enum):
    """Possible states for the pipeline."""
    STARTED = 0
    ENDED = 1
    PAUSED = 2
    ERROR = 3
    EMPTY = 4
    RUNNING = 5
    INITIALIZING = 6
    FORCED_STOP = 7


class IterationState(Enum):
    """Internal state to control the pipeline iterations."""
    COMPONENT_EXECUTED = 0
    COMPONENT_NOT_EXECUTED = 1
    COMPONENT_IN_EXECUTION = 2
