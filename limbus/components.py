from typing import Any, List
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import PIL
import torch
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

        self._outputs.declare("image")

    def forward(self, inputs: Params) -> ComponentState:
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
        self._outputs.image = image.clamp(0, 1)
        self._idx += 1
        return ComponentState.OK


class RGB2HLS(Component):
    """Component to add two input and output the result."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("inp")
        self._outputs.declare("out")

    def forward(self, inputs: Params) -> ComponentState:
        self._outputs.out = kornia.rgb_to_hls(inputs.inp)
        return ComponentState.OK


class Select(Component):
    """Component to add two input and output the result."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("c")
        self._inputs.declare("inp")
        self._outputs.declare("out")

    def forward(self, inputs: Params) -> ComponentState:
        self._outputs.out = torch.select(inputs.inp, -3, inputs.c)
        return ComponentState.OK


class ImageShow(Component):
    """Component to print the input in the console."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("image")

    def forward(self, inputs: Params) -> ComponentState:
        img = inputs.image
        if img.dim() == 2:
            img = img[None].repeat(3, 1, 1)
        img: np.ndarray = kornia.tensor_to_image(
            img.mul(255).clamp(0, 255).int())
        plt.figure(self._name)
        plt.imshow(img)
        plt.show()
        return ComponentState.OK


class Constant(Component):
    """Component that holds a constant."""
    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self._value = value
        self._outputs.declare("out")

    def forward(self, inputs: Params) -> ComponentState:
        self._outputs.out = self._value
        return ComponentState.OK


class Printer(Component):
    """Component to print the input in the console."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("inp")

    def forward(self, inputs: Params) -> ComponentState:
        print(inputs.inp)
        return ComponentState.OK


class Adder(Component):
    """Component to add two input and output the result."""
    def __init__(self, name: str):
        super().__init__(name)
        self._inputs.declare("a")
        self._inputs.declare("b")
        self._outputs.declare("sum_out")

    def forward(self, inputs: Params) -> ComponentState:
        self._outputs.sum_out = inputs.a + inputs.b
        return ComponentState.OK
