"""Module containing the basic functions to create components automatically."""

import inspect
from typing import Callable, Union, Dict, Any, cast

import typeguard
import torch.nn as nn

from ..core import Component, ComponentState, Params, NoValue


# the signature obtained from inspect.signature changes Optional to NoneType
NoneType = type(None)


def component_factory(callable_to_wrap: Union[Callable, type]) -> Component:
    """Generate a Component class for a given callable.

    Args:
        callable_to_wrap: callable to be wrapped as a component.

    Returns:
        Component wrapping the callable.

    """
    # torch functions are builtin, not standard functions
    if inspect.isfunction(callable_to_wrap) or inspect.isbuiltin(callable_to_wrap):
        return _component_func_factory(callable_to_wrap)

    if inspect.isclass(callable_to_wrap):
        assert isinstance(callable_to_wrap, type)
        if nn.Module not in inspect.getmro(callable_to_wrap):
            raise TypeError(f"{callable_to_wrap} does not inherit from nn.Module.")
        return _component_nn_factory(callable_to_wrap)
    raise TypeError(f"{callable_to_wrap} must be a function or an nn.Module.")


def _component_func_factory(callable_to_wrap: Callable) -> Component:
    """Generate a Component class for a given function.

    Args:
        callable_to_wrap: function to be wrapped as a component.

    Returns:
        Component wrapping the function.

    """
    # overwrite the forward(), register_inputs() and register_outputs() methods.
    def forward(self, inputs: Params) -> ComponentState:  # noqa: D417
        """Run the component.

        Args:
            inputs: set of values to be used to run the component.

        """
        args: Dict[str, Any] = {}
        for param in self._inputs.get_params():
            args[param] = inputs.get_param(param)

        res = callable_to_wrap(**args)

        if len(self._outputs.get_params()) > 1:
            for idx, param in enumerate(self._outputs.get_params()):
                self._outputs.set_param(param, res[idx])
        else:
            param = list(self._outputs.get_params())[0]
            self._outputs.set_param(param, res)
        return ComponentState.OK

    def register_inputs() -> Params:
        """Register the inputs params.

        Returns:
            input params

        """
        inputs = Params()
        sign: inspect.Signature = inspect.signature(callable_to_wrap)
        for param in sign.parameters.values():
            if param.default is param.empty:
                inputs.declare(param.name, param.annotation)
            else:
                inputs.declare(param.name, param.annotation, param.default)
        return inputs

    def register_outputs() -> Params:
        """Register the output params.

        Returns:
            output params

        """
        def isinstance_namedtuple(obj) -> bool:
            if typeguard.isclass(obj):
                return issubclass(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')
            return False
        outputs = Params()
        sign: inspect.Signature = inspect.signature(callable_to_wrap)
        if isinstance_namedtuple(sign.return_annotation):
            # variable number of outputs
            for k, v in sign.return_annotation._field_defaults.items():
                outputs.declare(k, v)
        else:
            if not typeguard.isclass(sign.return_annotation) and sign.return_annotation._name == "Tuple":
                # variable number of outputs
                for idx, arg in enumerate(sign.return_annotation.__args__):
                    outputs.declare(f"out{idx}", arg)
            else:
                # single output case
                outputs.declare("out", sign.return_annotation)
        return outputs

    str_name = f"{callable_to_wrap.__module__}.{callable_to_wrap.__name__}".replace(".", "___")
    return cast(Component, type(
        str_name, (Component,),
        {"forward": forward, "register_inputs": register_inputs, "register_outputs": register_outputs})
    )


def _component_nn_factory(callable_to_wrap: type) -> Component:
    """Generate a Component class for a given class.

    Args:
        callable_to_wrap: class to be wrapped as a component.

    Returns:
        Component wrapping the class.

    """
    # overwrite the forward(), register_inputs() and register_outputs() methods.
    def forward(self, inputs: Params) -> ComponentState:  # noqa: D417
        """Run the component.

        Args:
            inputs: set of values to be used to run the component.

        """
        args: Dict[str, Any] = {}
        for param in self._inputs.get_params():
            args[param] = inputs.get_param(param)

        # mypy cannot infer that the class has a forward method
        res = self._real_obj.forward(**args)  # type: ignore

        if len(self._outputs.get_params()) > 1:
            for idx, param in enumerate(self._outputs.get_params()):
                self._outputs.set_param(param, res[idx])
        else:
            param = list(self._outputs.get_params())[0]
            self._outputs.set_param(param, res)
        return ComponentState.OK

    def register_inputs() -> Params:
        """Register the inputs params.

        Returns:
            input params

        """
        inputs = Params()
        # mypy cannot infer that the class has a forward method
        sign: inspect.Signature = inspect.signature(callable_to_wrap.forward)  # type: ignore
        for param in sign.parameters.values():
            if param.name == "self":
                # skip the self parameter
                continue
            if param.default is param.empty:
                inputs.declare(param.name, param.annotation)
            else:
                inputs.declare(param.name, param.annotation, param.default)
        return inputs

    def register_outputs() -> Params:
        """Register the output params.

        Returns:
            output params

        """
        def isinstance_namedtuple(obj) -> bool:
            if typeguard.isclass(obj):
                return issubclass(obj, tuple) and hasattr(obj, '_asdict') and hasattr(obj, '_fields')
            return False
        outputs = Params()
        # mypy cannot infer that the class has a forward method
        sign: inspect.Signature = inspect.signature(callable_to_wrap.forward)  # type: ignore
        if isinstance_namedtuple(sign.return_annotation):
            # variable number of outputs
            for k, v in sign.return_annotation._field_defaults.items():
                outputs.declare(k, v)
        else:
            if not typeguard.isclass(sign.return_annotation) and sign.return_annotation._name == "Tuple":
                # variable number of outputs
                for idx, arg in enumerate(sign.return_annotation.__args__):
                    outputs.declare(f"out{idx}", arg)
            else:
                # single output case
                outputs.declare("out", sign.return_annotation)
        return outputs

    # convert signature parameters into Params
    sign: inspect.Signature = inspect.signature(callable_to_wrap)
    args = Params()
    for param in sign.parameters.values():
        if param.default is param.empty:
            args.declare(param.name, param.annotation)
        else:
            args.declare(param.name, param.annotation, param.default)

    # create an string containing all the parameters for the __init__ method as if they were written by hand
    str_params: str = ""
    str_args: str = ""
    for arg in args.get_params():
        if str_params != "":
            str_params += ", "
        if str_args != "":
            str_args += ", "
        str_args += arg
        value: Any = args.__getitem__(arg)  # get item don't raise an error if the type is not valid
        tp: type = args.get_type(arg)
        str_tp: str = str(tp)
        if typeguard.isclass(tp):  # if the type is a class, we need to remove <class '...'>
            str_tp = f"{tp.__module__}.{tp.__name__}"
        str_params += f"{arg}: {str_tp}"
        if not isinstance(value, NoValue):
            if isinstance(value, str):
                # if it is a string, we need to add quotes
                str_params += f" = '{value}'"
            else:
                str_params += f" = {value}"

    # template for the class to be created
    str_params = f"self, name: str, {str_params}"  # add the name parameter required by teh component
    str_name = f"{callable_to_wrap.__module__}.{callable_to_wrap.__name__}".replace(".", "___")
    func = (f"class {str_name}(Component):\n"
            f"    real_cls = {callable_to_wrap.__module__}.{callable_to_wrap.__name__}\n"
            f"    def __init__({str_params}) -> None:\n"
            f"        super().__init__(name)\n"
            f"        self._real_obj = {callable_to_wrap.__module__}.{callable_to_wrap.__name__}({str_args})\n")
    code = compile(func, __file__, "exec")
    eval(code, globals())
    globals()[str_name].forward = forward
    globals()[str_name].register_inputs = register_inputs
    globals()[str_name].register_outputs = register_outputs
    return globals()[str_name]
