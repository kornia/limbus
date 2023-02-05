import pytest
import asyncio

import torch

from limbus.core import Pipeline, PipelineState, VerboseMode, ComponentState
from limbus_components.base import Constant, Printer, Adder
from limbus_components.torch import Unbind


# TODO: test in detail the functions
@pytest.mark.usefixtures("event_loop_instance")
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
        c1.outputs.out.connect(unbind.inputs.tensor)
        c2.outputs.out.connect(unbind.inputs.dim)
        unbind.outputs.tensors.select(0).connect(show0.inputs.inp)
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
        assert pipeline.counter == 2

    async def test_pipeline_flow(self):
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
            assert c1.state == ComponentState.FORCED_STOP
            assert show0.state == ComponentState.FORCED_STOP
        await task()
        assert pipeline.counter > 0
        assert pipeline.counter < 5

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
