import pytest
import logging
import asyncio

import torch

from limbus.core import (Pipeline, PipelineState, VerboseMode, ComponentState, Component, OutputParams, OutputParam,
                         InputParam, async_utils)
from limbus_components.base import Constant, Printer, Adder
from limbus_components.torch import Unbind

log = logging.getLogger(__name__)


# TODO: test in detail the functions
class TestPipeline:
    def test_smoke(self):
        man = Pipeline()
        man is not None

    def test_pipeline(self):
        c1 = Constant("c1", 2 * torch.ones(1, 3))
        c2 = Constant("c2", torch.ones(1, 3))
        add = Adder("add")
        show = Printer("print")

        c1.outputs.out >> add.inputs.a
        c2.outputs.out >> add.inputs.b
        add.outputs.out >> show.inputs.inp

        pipeline = Pipeline()
        pipeline.add_nodes([c1, c2, add, show])
        out = pipeline.run(1)
        assert isinstance(out, PipelineState)

        torch.allclose(add.outputs.out.value, torch.ones(1, 3) * 3.)

    def test_pipeline_simple_graph(self):
        c1 = Constant("c1", torch.rand(2, 3))
        show0 = Printer("print0")
        c1.outputs.out.connect(show0.inputs.inp)
        pipeline = Pipeline()
        pipeline.add_nodes([c1, show0])
        out = pipeline.run(1)
        assert isinstance(out, PipelineState)

    def test_pipeline_disconnected_components(self):
        c1 = Constant("c1", torch.rand(2, 3))
        show0 = Printer("print0")
        c1.outputs.out.connect(show0.inputs.inp)
        c1.outputs.out.disconnect(show0.inputs.inp)
        pipeline = Pipeline()
        pipeline.add_nodes([c1, show0])
        out = pipeline.run(1)
        assert isinstance(out, PipelineState)

    def test_pipeline_iterable(self):
        c1 = Constant("c1", torch.rand(2, 3))
        c2 = Constant("c2", 0)
        unbind = Unbind("unbind")
        show0 = Printer("print0")
        c1.outputs.out.connect(unbind.inputs.input)
        c2.outputs.out.connect(unbind.inputs.dim)
        unbind.outputs.out.select(0).connect(show0.inputs.inp)
        pipeline = Pipeline()
        pipeline.add_nodes([c1, c2, unbind, show0])
        out = pipeline.run(1)
        assert isinstance(out, PipelineState)

    def test_pipeline_counter(self):
        c1 = Constant("c1", torch.rand(2, 3))
        show0 = Printer("print0")
        c1.outputs.out.connect(show0.inputs.inp)
        pipeline = Pipeline()
        pipeline.add_nodes([c1, show0])
        out = pipeline.run(2)
        assert isinstance(out, PipelineState)
        assert pipeline.min_iteration_in_progress == 2

    def test_pipeline_flow(self):
        c1 = Constant("c1", torch.rand(2, 3))
        show0 = Printer("print0")
        c1.outputs.out.connect(show0.inputs.inp)
        pipeline = Pipeline()
        pipeline.add_nodes([c1, show0])
        # tests before running
        assert pipeline._resume_event.is_set() is False
        pipeline.pause()
        assert pipeline._resume_event.is_set() is False
        pipeline.resume()
        assert pipeline._resume_event.is_set() is True
        pipeline.pause()

        # tests while running
        async def task():
            t = asyncio.create_task(pipeline.async_run())
            # wait for the pipeline to start (requires at least 2 iterations)
            await asyncio.sleep(0)
            assert pipeline._resume_event.is_set() is True
            assert pipeline._state.state == PipelineState.RUNNING
            pipeline.pause()
            assert pipeline._state.state == PipelineState.PAUSED
            assert pipeline._resume_event.is_set() is False
            assert pipeline._stop_event.is_set() is False
            await asyncio.sleep(0)
            pipeline.resume()
            assert pipeline._resume_event.is_set() is True
            # add some awaits to allow the pipeline to execute some components
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            pipeline.stop()
            assert pipeline._resume_event.is_set() is True
            assert pipeline._stop_event.is_set() is True
            assert pipeline.state == PipelineState.FORCED_STOP
            await asyncio.gather(t)
            assert len(c1.state) > 1
            assert ComponentState.FORCED_STOP in c1.state
            # Could be up to 3 states:
            # ComponentState.STOPPED_BY_COMPONENT
            # ComponentState.FORCED_STOP
            # E.g. ComponentState.OK
            assert c1.state_message(ComponentState.FORCED_STOP) is None
            assert len(show0.state) > 1
            assert ComponentState.FORCED_STOP in show0.state
            assert show0.state_message(ComponentState.FORCED_STOP) is None
        async_utils.run_coroutine(task())
        assert pipeline.min_iteration_in_progress > 0
        assert pipeline.min_iteration_in_progress < 5

    def test_pipeline_verbose(self):
        c1 = Constant("c1", torch.rand(2, 3))
        show0 = Printer("print0")
        c1.outputs.out.connect(show0.inputs.inp)
        pipeline = Pipeline()
        pipeline.add_nodes([c1, show0])
        assert pipeline._state.verbose == VerboseMode.DISABLED
        assert c1.verbose is False
        assert show0.verbose is False
        pipeline.set_verbose_mode(VerboseMode.COMPONENT)
        assert pipeline._state.verbose == VerboseMode.COMPONENT
        assert c1.verbose is True
        assert show0.verbose is True
        pipeline.set_verbose_mode(VerboseMode.PIPELINE)
        assert pipeline._state.verbose == VerboseMode.PIPELINE
        assert c1.verbose is False
        assert show0.verbose is False

    def my_testing_pipeline(self):
        class C(Component):
            @staticmethod
            def register_outputs(outputs: OutputParams) -> None:  # noqa: D102
                outputs.declare("out", int, arg="value")

            async def forward(self) -> ComponentState:  # noqa: D102
                if self.executions_counter == 2:
                    return ComponentState.STOPPED
                await self._outputs.out.send(1)
                return ComponentState.OK

        c1 = C("c1")
        show0 = Printer("print0")
        c1.outputs.out.connect(show0.inputs.inp)
        pipeline = Pipeline()
        pipeline.add_nodes([c1, show0])
        return pipeline

    def test_before_pipeline_user_hook(self, caplog):
        async def pipeline_hook(state: PipelineState):
            log.info(f"state: {state}")
        pipeline = self.my_testing_pipeline()
        pipeline.set_before_pipeline_user_hook(pipeline_hook)
        with caplog.at_level(logging.INFO):
            pipeline.run(1)
            assert "state: PipelineState.STARTED" in caplog.text

    def test_after_pipeline_user_hook(self, caplog):
        async def pipeline_hook(state: PipelineState):
            log.info(f"state: {state}")
        pipeline = self.my_testing_pipeline()
        pipeline.set_after_pipeline_user_hook(pipeline_hook)
        with caplog.at_level(logging.INFO):
            pipeline.run(3)
            assert "state: PipelineState.ENDED" in caplog.text

    def test_before_iteration_user_hook(self, caplog):
        async def iteration_hook(iter: int, state: PipelineState):
            log.info(f"iteration: {iter} ({state})")
        pipeline = self.my_testing_pipeline()
        pipeline.set_before_iteration_user_hook(iteration_hook)
        with caplog.at_level(logging.INFO):
            pipeline.run(1)
            assert "iteration: 1 (PipelineState.RUNNING)" in caplog.text

    def test_after_iteration_user_hook(self, caplog):
        async def iteration_hook(state: PipelineState):
            log.info(f"state: {state}")
        pipeline = self.my_testing_pipeline()
        pipeline.set_after_iteration_user_hook(iteration_hook)
        with caplog.at_level(logging.INFO):
            pipeline.run(1)
            assert "state: PipelineState.RUNNING" in caplog.text

    def test_before_component_user_hook(self, caplog):
        async def component_hook(cmp: Component):
            log.info(f"before component: {cmp.name}")
        pipeline = self.my_testing_pipeline()
        pipeline.set_before_component_user_hook(component_hook)
        with caplog.at_level(logging.INFO):
            pipeline.run(1)
            assert "before component" in caplog.text

    def test_after_component_user_hook(self, caplog):
        async def component_hook(cmp: Component):
            log.info(f"after component: {cmp.name}")
        pipeline = self.my_testing_pipeline()
        pipeline.set_after_component_user_hook(component_hook)
        with caplog.at_level(logging.INFO):
            pipeline.run(1)
            assert "after component" in caplog.text

    def test_param_received_user_hook(self, caplog):
        async def param_hook(param: InputParam):
            log.info(f"param: {param.name}")
        pipeline = self.my_testing_pipeline()
        pipeline.set_param_received_user_hook(param_hook)
        with caplog.at_level(logging.INFO):
            pipeline.run(1)
            assert "param: inp" in caplog.text

    def test_param_sent_user_hook(self, caplog):
        async def param_hook(param: OutputParam):
            log.info(f"param: {param.name}")
        pipeline = self.my_testing_pipeline()
        pipeline.set_param_sent_user_hook(param_hook)
        with caplog.at_level(logging.INFO):
            pipeline.run(1)
            assert "param: out" in caplog.text
