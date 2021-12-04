"""Some predefined components."""
from typing import Any, Callable, List, NamedTuple, Union, cast, Tuple, Dict, TypedDict, Optional, Literal, Sequence
from collections import namedtuple
from pathlib import Path
import hashlib
import logging
import inspect

from matplotlib import pyplot as plt
import numpy as np
import PIL
import torch
import visdom
import kornia

from limbus.core import Component, ComponentState, Params, component_factory

log = logging.getLogger(__name__)


class ExtraParams(TypedDict, total=False):
    """Typing for the arguments."""
    params: Dict[str, str]
    returns: Union[str, Dict[str, str], List[str]]
ComponentBuilder = Dict[str, ExtraParams]

# ways to declare a component:
# 1.- Params can be obtained with inspect and 1 output:
#     {"module.function": {}},
# 2.- Params can be obtained with inspect and several already typed outputs in a tuple:
#     {"module.function": {"returns": ["output_name_1", "output_name_2"]}},
# 3.- Params and returns can NOT be obtained with inspect:
#     {"module.function": {"params": {"input0": "typing0", "input1": "typing1",...},
#                          "returns": {"output0": "typing0"}}
# NOTE: second way also accepts typing ans in the third way.
lst_components: List[ComponentBuilder] = [
    {"kornia.enhance.image_histogram2d": {"returns": ["out", "out2"]}},  # we already know the types
    {"kornia.color.rgb_to_hls": {"returns": "torch.Tensor"}},
    {"kornia.color.hls_to_rgb": {}},
    {"kornia.enhance.equalize_clahe": {}},
    {"torch.select": {"params": {"input": "torch.Tensor", "dim": "int", "index": "int"},
                      "returns": {"out": "torch.Tensor"}}  # we do not know the types
    },
    {"torch.unbind": {"params": {"input": "torch.Tensor", "dim": "int"},
                      "returns": "Sequence[torch.Tensor]"}
    },
    {"torch.stack": {"params": {"input": "Sequence[torch.Tensor]", "dim": "int"},
                     "returns": {"out": "torch.Tensor"}}
    },
    {"torch.cat": {"params": {"input": "Sequence[torch.Tensor]", "dim": "int"},
                     "returns": {"out": "torch.Tensor"}}
    },
    {"torch.unsqueeze": {"params": {"input": "torch.Tensor", "dim": "int"},
                     "returns": {"out": "torch.Tensor"}}
    },
    {"torch.squeeze": {"params": {"input": "torch.Tensor", "dim": "Optional[int]"},
                     "returns": {"out": "torch.Tensor"}}
    }]

# TODO: add type checking when it is possible, validate that the number of input/outputs make sense...
def _create_ret_namedtuple(returns: Union[Dict[str, str], List[str]], tp: str, name: str) -> str:
    if isinstance(returns, list):
        named_tpl = (f"namedtuple('{tp}', returns,"
                     f"defaults=inspect.signature({name}).return_annotation.__args__)")
    else:
        named_tpl = f"namedtuple('{tp}', returns.keys(), defaults=list(map(eval, returns.values())))"
    globals()[tp] = eval(named_tpl)
    return tp

cmp: ComponentBuilder
for cmp in lst_components:
    for name, extras in cmp.items():
        fn_name = eval(name)
        str_name = name.replace(".", "___")
        params: Optional[Dict[str, str]] = extras.get("params", {})
        returns: Union[str, Dict[str, str], List[str]] = extras.get("returns", "")
        tp: str = f"{str_name}_ret"
        if not params:
            if returns:
                # NOTE: we are overriding the type of the return!!!
                if not isinstance(returns, str):
                    returns = _create_ret_namedtuple(returns, tp, name)
                fn_name.__annotations__["return"] = eval(returns)
            callable_function = fn_name
        else:
            if not isinstance(returns, str):
                # create namedtuple for the return values. THe default value denotes the type
                returns = _create_ret_namedtuple(returns, tp, name)
            # create wrapping function (e.g. torch methods do not have annotated typing)
            # NOTE: there is a trick to have acces to the name of the output parameters. The function signature
            # requires a namedtuple, however the return of the function is not a namedtuple.
            # Returning a namedtuple here is complex.
            str_params = str(params).replace("'","").replace("{","").replace("}","")
            str_ret_params = str(tuple(params.keys())).replace("'","")
            func = f"def {str_name}({str_params}) -> {returns}:\n    return real_func{str_ret_params}\n"
            code = compile(func, __file__, "exec")
            eval(code, {"real_func": fn_name}, globals())
            callable_function = globals()[str_name]

        globals()[str_name] = component_factory(callable_function)


class ImageReader(Component):
    """Component that holds a constant.

    Args:
        name: component name.
        value: path to an image or image folder.

    """
    def __init__(self, name: str, path: Path, batch_size: int = 1):
        super().__init__(name)
        self._value: List[Path] = []
        self._batch_size = batch_size
        self._idx = 0
        if Path(path).is_dir():
            self._value = sorted(list(Path(path).glob("*")))
        else:
            self._value.append(Path(path))

    @staticmethod
    def register_outputs() -> Params:  # noqa: D102
        outputs = Params()
        outputs.declare("image", torch.Tensor)
        return outputs
        
    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        images: List[torch.Tensor] = []
        batch_size = 0
        while batch_size < self._batch_size:
            if self._idx >= len(self._value):
                return ComponentState.STOPPED
            try:
                images.append(
                    kornia.image_to_tensor(np.asarray(PIL.Image.open(str(self._value[self._idx]))))
                )
                batch_size += 1
                self._idx += 1
            except:
                # avoid crashing the whole pipeline when there is a corrupted image or non-image file
                pass
        batch = torch.stack(images)
        # images must be in the range [0, 1]
        batch = batch.div(255.).clamp(0, 1)
        self._outputs.set_param("image", batch)
        return ComponentState.OK


class ImageShow(Component):
    """Component to show the input image."""
    def __init__(self, name: str):
        super().__init__(name)
        self._enabled = True
        try:
            self._visdom = visdom.Visdom(port=8087, raise_exceptions=True)
        except:
            self._enabled = False

        if self._enabled == False:
            log.warning("ImageShow is disabled!!!")
            return

        if not self._visdom.check_connection():
            raise ConnectionError('Error connecting with the visdom server.')

    @staticmethod
    def register_inputs() -> Params:  # noqa: D102
        inputs = Params()
        inputs.declare("image", torch.Tensor)
        return inputs

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        opts = {'title': self.name}
        # TODO: for batches use `images`
        images = inputs.get_param("image")
        if self._enabled:
            if images.shape[0] == 1:
                self._visdom.image(images[0], win=self.name, opts=opts)
            else:
                self._visdom.images(images, nrow=int(images.shape[0]), win=self.name, opts=opts)
            return ComponentState.OK
        # TODO: find a way to notify when the component is DISABLED or retruns an ERROR
        return ComponentState.DISABLED

    def finish_iter(self) -> None:  # noqa: D102
        pass


class Constant(Component):
    """Component that holds a constant."""
    def __init__(self, name: str, value: Any):
        super().__init__(name)
        self._value = value

    @staticmethod
    def register_outputs() -> Params:  # noqa: D102
        outputs = Params()
        outputs.declare("out", Any, arg="value")
        return outputs

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        # TODO: next line could be autogenerated fron the declare() method since there we are already linking both.
        self._outputs.set_param("out", self._value)
        return ComponentState.OK


class Printer(Component):
    """Component to print the input in the console."""
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def register_inputs() -> Params:  # noqa: D102
        inputs = Params()
        inputs.declare("inp", Any)
        return inputs

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        inp = inputs.get_param("inp")
        print(inp)
        return ComponentState.OK


# Example of a simple component created from the API
class Adder(Component):
    """Component to add two inputs and output the result."""
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def register_inputs() -> Params:  # noqa: D102
        inputs = Params()
        inputs.declare("a", torch.Tensor)
        inputs.declare("b", torch.Tensor)
        return inputs

    @staticmethod
    def register_outputs() -> Params:  # noqa: D102
        outputs = Params()
        outputs.declare("out", torch.Tensor)
        return outputs

    def forward(self, inputs: Params) -> ComponentState:  # noqa: D102
        a = inputs.get_param("a")
        b = inputs.get_param("b")
        self._outputs.set_param("out", a + b)
        return ComponentState.OK

