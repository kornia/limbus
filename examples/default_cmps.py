"""Basic example with predefined cmps."""
import asyncio
from sys import version_info
import copy

import torch

from limbus.core.pipeline import Pipeline
try:
    import limbus_components as components
except ImportError:
    raise ImportError("limbus-components is required to run this script."
                      "Install the package with: "
                      "'pip install limbus-components@git+https://git@github.com/kornia/limbus-components.git'")


# define your components
c1 = components.base.Constant("c1", 0)  # type: ignore
t1 = components.base.Constant("t1", torch.ones(1, 3))  # type: ignore
t2 = components.base.Constant("t2", torch.ones(1, 3) * 2)  # type: ignore
stack = components.torch.Stack("stack")  # type: ignore
show = components.base.Printer("print")  # type: ignore

# connect the components
c1.outputs.out >> stack.inputs.dim
t1.outputs.out >> stack.inputs.tensors.select(0)
t2.outputs.out >> stack.inputs.tensors.select(1)
stack.outputs.out >> show.inputs.inp

USING_PIPELINE = True
if USING_PIPELINE:
    # run your pipeline (only one iteration, note that this pipeline can run forever)
    print("Run with pipeline:")
    # create the pipeline and add its nodes
    pipeline = Pipeline()
    pipeline.add_nodes([c1, t1, t2, stack, show])
    pipeline.run(1)
else:
    # run 1 iteration using the asyncio loop
    print("Run with loop:")

    async def f():  # noqa: D103
        await asyncio.gather(c1(), t1(), t2(), stack(), show())

    if version_info.minor < 10:
        # for python <3.10 the loop must be run in this way to avoid creating a new loop.
        loop = asyncio.get_event_loop()
        loop.run_until_complete(f())
    elif version_info.minor >= 10:
        # for python >=3.10 the loop should be run in this way.
        asyncio.run(f())
