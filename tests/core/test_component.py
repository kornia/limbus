import pytest
import logging

from limbus.core import Component, ComponentState, Pipeline
from limbus.core.component import _ComponentState


class TestState:
    def test_smoke(self):
        cmp = Component("test")
        state = _ComponentState(cmp, ComponentState.RUNNING, True)
        assert state.state == [ComponentState.RUNNING]
        assert state.message(ComponentState.RUNNING) is None
        assert state.verbose is True
        assert state._component == cmp

    def test_call_no_params(self):
        cmp = Component("test")
        state = _ComponentState(cmp, ComponentState.RUNNING)
        assert state() == [ComponentState.RUNNING]
        assert state.message(ComponentState.RUNNING) is None
        assert state.verbose is False

    @pytest.mark.parametrize("verbose", [True, False])
    def test_call(self, caplog, verbose):
        cmp = Component("test")
        state = _ComponentState(cmp, ComponentState.RUNNING)
        state.verbose = verbose
        assert state.verbose == verbose
        with caplog.at_level(logging.INFO):
            assert state(ComponentState.DISABLED, "message") == [ComponentState.DISABLED]
            assert state.message(ComponentState.DISABLED) == "message"
            if verbose:
                assert len(caplog.records) == 1
                assert "message" in caplog.text
                assert caplog.records[0].levelname == "INFO"
            else:
                assert len(caplog.records) == 0


class TestComponent:
    def test_smoke(self):
        cmp = Component("yuhu")
        assert cmp.name == "yuhu"
        assert cmp.inputs is not None
        assert cmp.outputs is not None
        assert cmp.properties is not None

    def test_set_state(self):
        cmp = Component("yuhu")
        cmp.set_state(ComponentState.PAUSED)
        cmp.state == ComponentState.PAUSED
        cmp.state_message(ComponentState.PAUSED) is None
        cmp.state_message(ComponentState.FORCED_STOP) is None
        cmp.set_state(ComponentState.ERROR, "error")
        cmp.state == ComponentState.ERROR
        cmp.state_message(ComponentState.ERROR) == "error"
        cmp.state_message(ComponentState.FORCED_STOP) is None

    def test_set_properties(self):
        class A(Component):
            @staticmethod
            def register_properties(properties):
                Component.register_properties(properties)
                properties.declare("a", float, 1.)
                properties.declare("b", float, 2.)

        cmp = A("yuhu")
        assert cmp.properties.get_param("a") == 1.
        assert cmp.properties.get_param("b") == 2.
        assert cmp.properties.a() == 1.
        assert cmp.properties.b() == 2.
        cmp.properties.set_param("a", 3.)
        assert cmp.properties.get_param("a") == 3.
        assert cmp.set_properties(a=4., b=5.)
        assert cmp.properties.get_param("a") == 4.
        assert cmp.properties.get_param("b") == 5.
        assert cmp.set_properties(c=4.) is False
        p = cmp.properties.get_params()
        assert len(p) == 2
        assert p[0] in ["a", "b"]
        assert p[1] in ["a", "b"]
        p = cmp.properties.get_types()
        assert p["a"] == float
        assert p["b"] == float

    def test_register_inputs(self):
        class A(Component):
            @staticmethod
            def register_inputs(inputs):
                inputs.declare("a", float, 1.)
                inputs.declare("b", float, 2.)

        cmp = A("yuhu")
        assert len(cmp.outputs) == 0
        assert len(cmp.inputs) == 2
        assert cmp.inputs.a.value == 1.
        assert cmp.inputs.b.value == 2.

    def test_register_outputs(self):
        class A(Component):
            @staticmethod
            def register_outputs(outputs):
                outputs.declare("a", float, 1.)
                outputs.declare("b", float, 2.)

        cmp = A("yuhu")
        assert len(cmp.inputs) == 0
        assert len(cmp.outputs) == 2
        assert cmp.outputs.a.value == 1.
        assert cmp.outputs.b.value == 2.

    def test_init_from_component(self):
        class A(Component):
            pass

        cmp = A("yuhu")
        cmp.verbose is True
        cmp2 = A("yuhu2")
        assert cmp2.verbose is False
        cmp2.init_from_component(cmp)
        assert cmp2.verbose == cmp.verbose
        assert cmp2.pipeline == cmp.pipeline  # None


@pytest.mark.usefixtures("event_loop_instance")
class TestComponentWithPipeline:
    def test_init_from_component_with_pipeline(self):
        class A(Component):
            @staticmethod
            def register_outputs(outputs):
                outputs.declare("out", int)

            async def forward():
                self._outputs.out.send(1)
                return ComponentState.OK

        class B(Component):
            @staticmethod
            def register_inputs(inputs):
                inputs.declare("inp", int)

            async def forward():
                self._inputs.inp.receive()
                return ComponentState.OK

        a = A("a")
        b = B("b")
        a.outputs.out >> b.inputs.inp
        pipeline = Pipeline()
        pipeline.add_nodes(a)
        assert len(pipeline._nodes) == 1
        assert a in pipeline._nodes
        a.set_pipeline(pipeline)
        a.verbose = True
        b.init_from_component(a)
        assert a.verbose == b.verbose
        assert a.pipeline == b.pipeline  # None
        assert len(pipeline._nodes) == 2
        assert a in pipeline._nodes
        assert b in pipeline._nodes

    @pytest.mark.parametrize("iters", [0, 1, 2])
    def test_get_stopping_iteration(self, iters):
        class A(Component):
            async def forward(self):
                return ComponentState.STOPPED

        cmp = A("yuhu")
        pipeline = Pipeline()
        pipeline.add_nodes(cmp)
        pipeline.run(iters)
        assert cmp.executions_counter == 1
        assert pipeline.min_iteration_in_progress == 1
        assert cmp.stopping_execution == iters

    def test_stop_after_exception(self):
        class A(Component):
            async def forward(self):
                raise Exception("test")

        cmp = A("yuhu")
        pipeline = Pipeline()
        pipeline.add_nodes(cmp)
        pipeline.run(2)
        assert cmp.executions_counter == 1
        assert pipeline.min_iteration_in_progress == 1
        assert cmp.state == [ComponentState.ERROR]
        assert cmp.state_message(ComponentState.ERROR) == "Exception - test"

    def test_stop_after_stop(self):
        class A(Component):
            async def forward(self):
                return ComponentState.STOPPED

        cmp = A("yuhu")
        pipeline = Pipeline()
        pipeline.add_nodes(cmp)
        pipeline.run(2)
        assert cmp.executions_counter == 1
        assert pipeline.min_iteration_in_progress == 1
        assert cmp.state == [ComponentState.STOPPED]
        assert cmp.state_message(ComponentState.STOPPED) is None
