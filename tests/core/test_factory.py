import limbus.components
from limbus.core import Component, ComponentState, Params, register_component


def test_registry():
    # Example of a simple component created from the API
    @register_component
    class TestComp(Component):
        """Component to add two inputs and output the result."""
        def __init__(self, name: str):
            super().__init__(name)

        @staticmethod
        def register_inputs() -> Params:  # noqa: D102
            inputs = Params()
            inputs.declare("a", int)
            return inputs

        @staticmethod
        def register_outputs() -> Params:  # noqa: D102
            outputs = Params()
            outputs.declare("out", int)
            return outputs

        def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
            a = inputs.get_param("a")
            self._outputs.set_param("out", a)
            return ComponentState.OK

    assert limbus.components.test_factory___TestComp('test')  # type: ignore
