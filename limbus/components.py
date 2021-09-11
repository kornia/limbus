"""Some predefnied components."""
from typing import Any, List, cast, Tuple
from pathlib import Path
import hashlib

from matplotlib import pyplot as plt
import numpy as np
import PIL
import torch
from torch.utils.tensorboard import SummaryWriter
import visdom
import kornia

from limbus.core import Component, ComponentState, Params


class ImageReader(Component):
    """Component that holds a constant.

    Args:
        name: component name.
        value: path to an image or image folder.

    """
    def __init__(self, name: str, value: str):
        super().__init__(name)
        self._value: List[Path] = []
        self._idx = 0
        if Path(value).is_dir():
            self._value = list(Path(value).glob("*"))
        else:
            self._value.append(Path(value))

        self._outputs.declare("image", torch.Tensor)
        self._outputs.declare("name", str)
        self._outputs.declare("counter", str)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        while True:
            if self._idx >= len(self._value):
                return ComponentState.STOPPED
            try:
                image: torch.Tensor = kornia.image_to_tensor(np.asarray(PIL.Image.open(str(self._value[self._idx]))))
                break
            except:
                self._idx += 1
        # images must be in the range [0, 1]
        image = image.div(255.)
        self._outputs.set_param("image", image.clamp(0, 1))
        self._outputs.set_param("name", str(self._value[self._idx].name))
        self._outputs.set_param("counter", f"{self._idx} / {len(self._value)}")
        self._idx += 1
        return ComponentState.OK


class RGB2HLS(Component):
    """Component to convert a rgb image to hls."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("inp", torch.Tensor)
        self._outputs.declare("out", torch.Tensor)

    def forward(self, inputs) -> ComponentState:  # noqa: D102
        inp = inputs.get_param("inp")
        self._outputs.set_param("out", kornia.rgb_to_hls(inp))
        return ComponentState.OK


class HLS2RGB(Component):
    """Component to convert a hls image to rgb."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("inp", torch.Tensor)
        self._outputs.declare("out", torch.Tensor)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        inp = inputs.get_param("inp")
        self._outputs.set_param("out", kornia.hls_to_rgb(inp))
        return ComponentState.OK


class Select(Component):
    """Component to select one channels of an image."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("c", int)
        self._inputs.declare("inp", torch.Tensor)
        self._outputs.declare("out", torch.Tensor)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        inp = inputs.get_param("inp")
        c = inputs.get_param("c")
        self._outputs.set_param("out", torch.select(inp, -3, c))
        return ComponentState.OK


class Unbind(Component):
    """Component to unbind one tensor."""
    def __init__(self, name: str, value: int):
        super().__init__(name)
        self._value = value
        self._inputs.declare("inp", torch.Tensor)
        for v in range(value):
            self._outputs.declare(str(v), torch.Tensor)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        inp = inputs.get_param("inp")
        out: List[torch.Tensor] = cast(List[torch.Tensor], torch.unbind(inp, -3))
        for idx, v in enumerate(out):
            # NOTE: we are not controlling if we are adding more pins
            self._outputs.set_param(str(idx), v)
        return ComponentState.OK


class Stack(Component):
    """Component to stack tensors."""
    def __init__(self, name: str, value: int):
        super().__init__(name)
        self._value = value
        for v in range(value):
            self._inputs.declare(str(v), torch.Tensor)
        self._outputs.declare("out", torch.Tensor)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        tensors: List[torch.Tensor] = [inputs[str(idx)] for idx in range(self._value)]
        self._outputs.set_param("out", torch.stack(tensors))
        return ComponentState.OK


class Clahe(Component):
    """Component to apply clahe and output the result."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("inp", kornia.enhance.equalize_clahe.__annotations__["input"])
        self._inputs.declare("clip_limit", kornia.enhance.equalize_clahe.__annotations__["clip_limit"], 2.)
        self._inputs.declare("grid_size", kornia.enhance.equalize_clahe.__annotations__["grid_size"], (8, 8))
        self._outputs.declare("out", torch.Tensor)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        inp = inputs.get_param("inp")
        clip_limit = inputs.get_param("clip_limit")
        grid_size = inputs.get_param("grid_size")
        self._outputs.set_param("out", (
            kornia.enhance.equalize_clahe(inp[None], clip_limit, grid_size)).squeeze_(0)
        )
        return ComponentState.OK


class ImageShow(Component):
    """Component to show the input image."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("image", torch.Tensor)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        image = inputs.get_param("image")
        if image.dim() == 2:
            image = image[None].repeat(3, 1, 1)
        img: np.ndarray = kornia.tensor_to_image(
            image.mul(255).clamp(0, 255).int())
        fig = plt.figure(int(hashlib.md5(self._name.encode()).hexdigest(), 16))
        fig.canvas.set_window_title(self._name)
        fig.add_subplot(111).imshow(img)
        plt.show(block=False)
        return ComponentState.OK

    def finish_iter(self) -> None:  # noqa: D102
        plt.show()


class ImageShowTensorboard(Component):
    """Component to show the input image."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("image", torch.Tensor)
        self._writer = SummaryWriter('tensorboard')

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        image = inputs.get_param("image")
        self._writer.add_image(self.name, image)
        return ComponentState.OK

    def finish_iter(self) -> None:  # noqa: D102
        self._writer.flush()

    def __del__(self):
        self._writer.close()


class ImageShow_(Component):
    """Component to show the input image."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("image", torch.Tensor)
        self._visdom = visdom.Visdom(port=8087, raise_exceptions=True)
        if not self._visdom.check_connection():
            raise ConnectionError('Error connecting with the visdom server.')

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        opts = {'title': self.name}
        # TODO: for batches use `images`
        image = inputs.get_param("image")
        self._visdom.image(image, win=self.name, opts=opts)
        return ComponentState.OK

    def finish_iter(self) -> None:  # noqa: D102
        pass


class Constant(Component):
    """Component that holds a constant."""
    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self._value = value
        self._outputs.declare("out", object)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        self._outputs.set_param("out", self._value)
        return ComponentState.OK


class Printer(Component):
    """Component to print the input in the console."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("inp", object)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        inp = inputs.get_param("inp")
        print(inp)
        return ComponentState.OK


class Adder(Component):
    """Component to add two inputs and output the result."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("a", torch.Tensor)
        self._inputs.declare("b", torch.Tensor)
        self._outputs.declare("out", torch.Tensor)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        a = inputs.get_param("a")
        b = inputs.get_param("b")
        self._outputs.set_param("out", a + b)
        return ComponentState.OK


class Multiplier(Component):
    """Component to multiply two inputs and output the result."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("a", torch.Tensor)
        self._inputs.declare("b", torch.Tensor)
        self._outputs.declare("out", torch.Tensor)

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        a = inputs.get_param("a")
        b = inputs.get_param("b")
        self._outputs.set_param("out", a * b)
        return ComponentState.OK
