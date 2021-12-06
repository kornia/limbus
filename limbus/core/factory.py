"""Module containing the basic functions to create components automatically."""

import inspect
from typing import Callable, Union, Dict, Any, cast, List, TypedDict, Optional, NamedTuple, Tuple
import importlib

import typeguard
import torch.nn as nn

from limbus.core import Component, ComponentState, Params, NoValue


# the signature obtained from inspect.signature changes Optional to NoneType
NoneType = type(None)


# define ComponentBuilder which is the structure containing the required data to build components automatically
class ExtraParams(TypedDict, total=False):
    """Typing for the arguments."""
    params: Dict[str, str]
    returns: Union[str, Dict[str, str], List[str]]


ComponentDefinition = Dict[str, ExtraParams]


def _add_modules_to_globals(comp_globals: Dict[str, Any], modules: List[str]) -> None:
    """Add the modules in the list to the globals where the component will be defined."""
    for module in modules:
        if module.find(".") != -1:
            module = module[0: module.find(".")]
        if module not in comp_globals:
            comp_globals[module] = importlib.import_module(module)


def _get_annotation(annotation: Any, comp_globals: Dict[str, Any]) -> str:
    """Get the string representation of the type of a parameter and add the module of the type to the globals."""
    # if it is a standard type...
    if typeguard.isclass(annotation):
        # add the module where the type is defined to ensure it is accesible.
        _add_modules_to_globals(comp_globals, [annotation.__module__])
        return f"{annotation.__module__}.{annotation.__name__}"
    # else we assume it is a typing expresion...
    else:
        # TODO: check if inside the typing expression the base types are always defined within module
        _add_modules_to_globals(comp_globals, ["typing"])
        return str(annotation)


def _get_params(name: str, comp_globals: Dict[str, Any]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    _add_modules_to_globals(comp_globals, ["inspect"])
    sign = eval(f"inspect.signature({name})", comp_globals)
    for param in sign.parameters.values():
        params[param.name] = _get_annotation(param.annotation, comp_globals)
        # if there is a default value, we add it
        if param.default is not inspect.Parameter.empty:
            default: Any = param.default
            if isinstance(param.default, str):
                default = f"\"{param.default}\""
            elif type(param.default).__module__ != "builtins":
                # in this case we concat the module to be able to retrieve the value.
                # TODO: this trick is not gonan work always.
                default = f"{type(param.default).__module__}.{param.default}"
            params[param.name] += f" = {default}"
    return params


def _get_params_as_def(params: Dict[str, str]) -> Tuple[str, str]:
    """Return the params formated as in a function definition and as params to be passed to a funciton."""
    # convert the params to a string
    str_params_def: str = ""
    str_params: str = ""
    for param_name, param_typing in params.items():
        str_params_def += f"{param_name}: {param_typing}, "
        str_params += f"{param_name}, "
    if str_params_def:
        str_params_def = str_params_def[:-2]
        str_params = str_params[:-2]
    return str_params_def, str_params


def _build_returns(returns: Union[str, Dict[str, str], List[str]],
                   tp: Optional[str],
                   name: Optional[str],
                   comp_globals: Dict[str, Any]) -> str:
    """Build the return of the component based on the signature of the callable.

    Warning: If we force any return type and it is wrong we are not raising an error!!!

    Args:
        returns: type of the return of the component. Can be defined as:
            - "" if we want the default return of the callable
            - "type" to force the return type of the callable
            - List[str] to force the name of the output parameter of the callable
            - Dict[str, str] to force the name and the type of the output parameter of the callable
        tp: name of the namedtuple to be returned (required if returns != str).
        name: name of the callable (required if returns != str).
        comp_globals: globals of the module where the component will be defined.

    Returns:
        str denoting the build return type

    """
    if isinstance(returns, str):
        # the return type is defined in the string
        if returns == "":
            returns = eval(f"inspect.signature({name}).return_annotation", comp_globals)
        return _get_annotation(returns, comp_globals)
    else:
        # the return type will a namedtuple
        assert tp is not None
        assert name is not None
        if isinstance(returns, list):
            _add_modules_to_globals(comp_globals, ["collections", "inspect"])
            named_tpl = (f"collections.namedtuple('{tp}', {returns},"
                         f"defaults=inspect.signature({name}).return_annotation.__args__)")
        else:
            _add_modules_to_globals(comp_globals, ["collections"])
            named_tpl = (f"collections.namedtuple('{tp}', {returns}.keys(),"
                         f"defaults=list(map(eval, {returns}.values())))")
        # the new type must be in the globals to be used
        comp_globals[tp] = eval(named_tpl, comp_globals)
        return tp


def register_components(comp_globals: Dict[str, Any], lst_components: List[ComponentDefinition]) -> None:
    """Register all teh components of a list of components.

    Args:
        comp_globals: globals() of the module where the components will be defined.
        lst_components: List of components to register.

    """
    cmp: ComponentDefinition
    for cmp in lst_components:
        # get info from the component definition
        name: str = list(cmp.keys())[0]
        extras: ExtraParams = list(cmp.values())[0]
        params: Dict[str, str] = extras.get("params", {})
        returns: Union[str, Dict[str, str], List[str]] = extras.get("returns", "")

        # add the base module in "name" to the globals if it is not there
        _add_modules_to_globals(comp_globals, [name])

        # if it is a class we directly try to create the component
        fn_name: Union[Callable, type] = eval(name, comp_globals)
        if inspect.isclass(fn_name):
            component_factory(name, fn_name, comp_globals)
        # else we create a wrapper that will be used to create the component. We need that wrapper to deal
        # with the pytorch functions that do not have typing.
        else:
            # the name of the component will be the original callable name but replacing . by ___
            str_name = name.replace(".", "___")
            # the component return is a namedtuple with the same name + _ret
            tp: str = f"{str_name}_ret"

            # if params are not defined, they are obtained from the signature of the callable
            if not params:
                params = _get_params(name, comp_globals)

            return_type: Union[str, NamedTuple] = _build_returns(returns, tp, name, comp_globals)

            # create wrapping code for the callable
            # -------------------------------------
            # WARNING: the wrapping code specifies a return type that can be different from the return type of
            # the callable. However, both must be compatible. If the return type of the callable is not compatible
            # an error will be raised in execution time. To be able to assign names to the outputs we needed to convert
            # them into a namedtuple.

            # convert the params to a string
            str_params_def, str_params = _get_params_as_def(params)
            # define and compile the code of the wrapped callable
            func = f"def {str_name}({str_params_def}) -> {return_type}:\n    return real_func({str_params})\n"
            code = compile(func, comp_globals["__file__"], "exec")
            eval(code, {"real_func": fn_name}, comp_globals)
            # create the component from the callable
            component_factory(name, comp_globals[str_name], comp_globals)


# define the factory functions to create components automatically
def component_factory(name: str, callable_to_wrap: Union[Callable, type], comp_globals: Dict[str, Any]) -> None:
    """Generate a Component class for a given callable and add it to the globals.

    Args:
        name: name of the function to be wrapped as a component.
        callable_to_wrap: callable to be wrapped as a component.
        comp_globals: globals() of the module where the components will be defined.

    """
    # torch functions are builtin, not standard functions
    if inspect.isfunction(callable_to_wrap) or inspect.isbuiltin(callable_to_wrap):
        return _component_func_factory(name, callable_to_wrap, comp_globals)

    if inspect.isclass(callable_to_wrap):
        assert isinstance(callable_to_wrap, type)
        if nn.Module not in inspect.getmro(callable_to_wrap):
            raise TypeError(f"{callable_to_wrap} does not inherit from nn.Module.")
        return _component_nn_factory(name, callable_to_wrap, comp_globals)
    raise TypeError(f"{callable_to_wrap} must be a function or an nn.Module.")


def _component_func_factory(name: str, callable_to_wrap: Callable, comp_globals: Dict[str, Any]) -> None:
    """Generate a Component class for a given function and add it to the globals.

    Args:
        name: name of the function to be wrapped as a component.
        callable_to_wrap: function to be wrapped as a component.
        comp_globals: globals() of the module where the components will be defined.

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

    # add the NoneType type to the globals.
    # NOTE: This type is returned by inspect.signature but it does not exist
    comp_globals['NoneType'] = NoneType

    str_name = name.replace(".", "___")
    comp_globals[str_name] = cast(Component, type(
        str_name, (Component,),
        {"forward": forward, "register_inputs": register_inputs, "register_outputs": register_outputs})
    )


def _component_nn_factory(name: str, callable_to_wrap: type, comp_globals: Dict[str, Any]) -> None:
    """Generate a Component class for a given class and add it to the globals.

    Args:
        name: name of the function to be wrapped as a component.
        callable_to_wrap: class to be wrapped as a component.
        comp_globals: globals() of the module where the components will be defined.

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

    # convert signature parameters into a set of params
    params: Dict[str, str] = _get_params(name, comp_globals)

    # create an string containing all the parameters for the __init__ method as if they were written by hand
    str_params_def, str_params = _get_params_as_def(params)

    # add the NoneType type to the globals.
    # NOTE: This type is returned by inspect.signature but it does not exist
    comp_globals['NoneType'] = NoneType

    # template for the class to be created
    str_params_def = f"self, name: str, {str_params_def}"  # add the name parameter required by teh component
    str_name = name.replace(".", "___")
    func = (f"class {str_name}(Component):\n"
            f"    real_cls = {callable_to_wrap.__module__}.{callable_to_wrap.__name__}\n"
            f"    def __init__({str_params_def}) -> None:\n"
            f"        super().__init__(name)\n"
            f"        self._real_obj = {callable_to_wrap.__module__}.{callable_to_wrap.__name__}({str_params})\n")
    code = compile(func, comp_globals["__file__"], "exec")
    eval(code, comp_globals)
    comp_globals[str_name].forward = forward
    comp_globals[str_name].register_inputs = register_inputs
    comp_globals[str_name].register_outputs = register_outputs
