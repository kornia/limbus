"""Basic example with predefined cmps."""
import asyncio

import torch

from limbus.core.pipeline import Pipeline
try:
    import limbus_components as components
except ImportError:
    raise ImportError("limbus-components is required to run this script. Reinstall limbus with: "
                      "'pip install limbus[components]' or install the package with: "
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
stack.outputs.tensor >> show.inputs.inp

# create the pipeline and add its nodes
pipeline = Pipeline()
pipeline.add_nodes([c1, t1, t2, stack, show])

# run your pipeline (only one iteration, note that this pipeline can run forever)
print("Run with pipeline:")
pipeline.run(1)

# run 1 iteration using the asyncio loop
print("Run with loop:")
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(c1(), t1(), t2(), stack(), show()))
