"""Module containing the basic functions to create components automatically."""

import inspect
from typing import Callable, Union, Dict, Any, cast, List, TypedDict, Optional, NamedTuple, Tuple
import importlib
import logging

import yaml
import typeguard
import torch.nn as nn

from limbus.core import Component, ComponentState, Params, NoValue


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# NOTE: this globals must be init before running anything!!!
# they are init from the module where the components will be added.
COMP_GLOBALS: Dict[str, Any] = {}

# the signature obtained from inspect.signature changes Optional to NoneType
NoneType = type(None)


# define ComponentDefinition which is the structure containing the required data to build components automatically
class ExtraParams(TypedDict, total=False):
    """Typing for the arguments."""
    params: Dict[str, str]
    returns: Union[str, Dict[str, str], List[str]]


ComponentDefinition = Dict[str, ExtraParams]


def component_factory(name: str, callable_to_wrap: Union[Callable, type], extra: ExtraParams) -> None:
    """Generate a Component class for a given callable/nn.Module and add it to the globals.

    Args:
        name: name of the function to be wrapped as a component.
        callable_to_wrap: callable to be wrapped as a component.
        extra: extra parameters used to create the component.

    """
    # add the NoneType type to the globals.
    # NOTE: This type is returned by inspect.signature but it does not exist
    COMP_GLOBALS['NoneType'] = NoneType

    # ATTENTION: In this function "callable_forward" var is used as pointer to change
    # the job done inside the new component class. The register_inputs() and register_outputs() methods use directly
    # this var to get the input and output params. The forward() is a bit trickier since the self._callable_forward
    # var is assigned inside the template of the component class.
    callable_forward: Union[Callable, type]

    # 1. define those vars
    # --------------------
    # torch functions are builtin and their signatures mast be imported in a special way
    is_torch: bool = (name[0: name.find(".")] == "torch")
    # We need to distinguish between functions and nn.Modules.
    if inspect.isfunction(callable_to_wrap) or (is_torch and inspect.isbuiltin(callable_to_wrap)):
        # in this case the "callable_forward" is directly the callable
        callable_forward = callable_to_wrap
    elif isinstance(callable_to_wrap, type) and nn.Module in inspect.getmro(callable_to_wrap):
        # in this case "callable_forward" refers to the forward method in the original nn.Module
        callable_forward = callable_to_wrap.forward  # type: ignore  # mypy don't know about the forward method
    else:
        raise TypeError(f"{callable_to_wrap} must be a function, a torch builtin or an nn.Module.")

    # 2. define the forward(), register_inputs() and register_outputs() methods
    # -------------------------------------------------------------------------
    def forward(self, inputs: Params) -> ComponentState:  # noqa: D417
        """Run the component.

        Args:
            inputs: set of values to be used to run the component.

        """
        args: Dict[str, Any] = {}
        for param in self._inputs.get_params():
            args[param] = inputs.get_param(param)
        # mypy cannot infer that the class has this method
        res = self._callable(**args)  # type: ignore
        if len(self._outputs.get_params()) > 1:
            for idx, param in enumerate(self._outputs.get_params()):
                self._outputs.set_param(param, res[idx])
        else:
            param = list(self._outputs.get_params())[0]
            self._outputs.set_param(param, res)
        return ComponentState.OK

    def _import_if_it_is_possible(modules: List[str]) -> None:
        # if there is a module to import, let's import it
        try:
            _add_modules_to_globals(modules)
        except:
            pass

    def register_inputs() -> Params:
        """Register the inputs params.

        Returns:
            input params

        """
        inputs = Params()
        params: Union[Dict[str, str], List[inspect.Parameter]] = extra.get("params", {})
        if not params:
            # NOTE: callable_forward needs to be predefined
            # if there are no params in extra, we use the signature of the function
            sign: inspect.Signature = inspect.signature(callable_forward)
            params = list(sign.parameters.values())
        else:
            # just in case it is not imported and there are typing types in the params
            _add_modules_to_globals(["typing"])
        for param in params:
            if isinstance(param, inspect.Parameter):
                p_name: str = param.name
                p_annotation: Any = param.annotation
                p_default: Any = param.default
                p_empty: bool = p_default is param.empty
            else:
                assert isinstance(params, dict)
                p_name = param
                p_annotation = params[param]
                p_empty = p_annotation.find("=") == -1
                if not p_empty:
                    p_default = p_annotation[p_annotation.find("=") + 1:]
                    p_annotation = p_annotation[0: p_annotation.find("=")]
                _import_if_it_is_possible([p_annotation])
                # convert to the proper type the type and default value
                p_default = eval(p_default, COMP_GLOBALS) if not p_empty else None
                p_annotation = eval(p_annotation, COMP_GLOBALS)
            # skip the self parameter
            if p_name == "self":
                continue
            if p_empty:
                inputs.declare(p_name, p_annotation)
            else:
                inputs.declare(p_name, p_annotation, p_default)
        return inputs

    def _helper_to_add_returns(outputs: Params, return_annotation: Any, name: Optional[List[str]] = None) -> None:
        if name is None:
            name = ["out"]
        if not typeguard.isclass(return_annotation) and return_annotation._name == "Tuple":
            # variable number of outputs
            # NOTE: if there are several names, len(name) and the number of returns must coincide!!!
            for idx, arg in enumerate(return_annotation.__args__):
                out_name = f"{name[0]}{idx}" if len(name) == 1 else f"{name[idx]}"
                outputs.declare(out_name, arg)
        else:
            # single output case
            outputs.declare(f"{name[0]}", return_annotation)

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
        returns: Union[str, Dict[str, str], List[str]] = extra.get("returns", {})
        # if the returns are a list we still need the signature of the function
        if not returns or isinstance(returns, list):
            # NOTE: callable_forward needs to be predefined
            # if there are no returns in extra, we use the signature to get them
            sign: inspect.Signature = inspect.signature(callable_forward)
            return_annotation: Any = sign.return_annotation
        # get returns only using the signature
        if not returns:
            if isinstance_namedtuple(return_annotation):
                # variable number of outputs
                for k, v in return_annotation._field_defaults.items():
                    outputs.declare(k, v)
            else:
                _helper_to_add_returns(outputs, return_annotation)

        #############################################################################################
        # Valid formats for teh "returns" variable:
        #    - "" if we want the default return of the callable
        #    - "type" to force the return type of the callable
        #    - List[str] to force the name of the output parameter of the callable
        #    - Dict[str, str] to force the name and the type of the output parameter of the callable
        #############################################################################################
        # if returns is a string we only need to eval() it
        elif isinstance(returns, str):
            _import_if_it_is_possible([returns])
            outputs.declare("out", eval(returns, COMP_GLOBALS))
        # if returns is a list we need to rename the outputs with the names in the list
        elif isinstance(returns, list):
            _helper_to_add_returns(outputs, return_annotation, returns)
        elif isinstance(returns, dict):
            _add_modules_to_globals(["typing"])
            ret_name: List[str] = []
            return_annotation = ""
            for key in returns:
                ret_name.append(key)
                _import_if_it_is_possible([returns[key]])
                return_annotation += f"{returns[key]}, "
            return_annotation = return_annotation[:-2]
            if len(ret_name) > 1:
                return_annotation = f"typing.Tuple[{return_annotation}]"
            _helper_to_add_returns(outputs, eval(return_annotation, COMP_GLOBALS), ret_name)
        else:
            raise TypeError("Invalid type definition for the output pins.")
        return outputs

    # 3. create the component class
    # -----------------------------
    # define the name of hte component class
    str_name = name.replace(".", "___")
    # create the class
    if inspect.isfunction(callable_to_wrap) or inspect.isbuiltin(callable_to_wrap):
        # 1. define the class template
        func = (f"class {str_name}(Component):\n"
                f"    def __init__(self, name: str) -> None:\n"
                f"        super().__init__(name)\n"
                f"        self._callable = callable_forward\n")
        # 2. compile and change the keyword "callable_forward" for the real function to be run
        code = compile(func, COMP_GLOBALS["__file__"], "exec")
        eval(code, {"callable_forward": callable_forward}, COMP_GLOBALS)
    else:
        # In the case of an nn.Module we need to dinamically assign the params to the __init__ method and
        # create the original object.
        # 1. Write the parameters of the __init__ method and the call to the forward method
        params: Dict[str, str] = _get_params(name)
        str_params_def, str_params = _get_params_as_def(params)

        # 2. write the template for the component
        str_params_def = f"self, name: str, {str_params_def}"  # add the name parameter required by the component
        func = (f"class {str_name}(Component):\n"
                f"    def __init__({str_params_def}) -> None:\n"
                f"        super().__init__(name)\n"
                f"        self._real_obj = {name}({str_params})\n"
                f"        self._callable = self._real_obj.forward\n")
        # 3. compile the code and add the methods to the component class
        code = compile(func, COMP_GLOBALS["__file__"], "exec")
        eval(code, COMP_GLOBALS)
    COMP_GLOBALS[str_name].forward = forward
    COMP_GLOBALS[str_name].register_inputs = register_inputs
    COMP_GLOBALS[str_name].register_outputs = register_outputs


def _add_modules_to_globals(modules: List[str]) -> None:
    """Add the modules in the list to the globals where the component will be defined."""
    for module in modules:
        if module.find(".") != -1:
            module = module[0: module.find(".")]
        if module not in COMP_GLOBALS:
            COMP_GLOBALS[module] = importlib.import_module(module)


def _get_annotation(annotation: Any) -> str:
    """Get the string representation of the type of a parameter and add the module of the type to the globals."""
    # if it is a standard type...
    if typeguard.isclass(annotation):
        # add the module where the type is defined to ensure it is accesible.
        _add_modules_to_globals([annotation.__module__])
        return f"{annotation.__module__}.{annotation.__name__}"
    # else we assume it is a typing expresion...
    else:
        # TODO: check if inside the typing expression the base types are always defined within module
        _add_modules_to_globals(["typing"])
        return str(annotation)


def _get_params(name: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    _add_modules_to_globals(["inspect"])
    sign = eval(f"inspect.signature({name})", COMP_GLOBALS)
    for param in sign.parameters.values():
        params[param.name] = _get_annotation(param.annotation)
        # if there is a default value, we add it
        if param.default is not inspect.Parameter.empty:
            default: Any = param.default
            if isinstance(param.default, str):
                default = f"\"{param.default}\""
            elif type(param.default).__module__ != "builtins":
                # in this case we concat the module to be able to retrieve the value.
                # TODO: this trick is not gonna work always.
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


def register_components(lst_components: List[ComponentDefinition]) -> None:
    """Register all the components of a list of components.

    Args:
        lst_components: List of components to register.

    """
    cmp: ComponentDefinition
    for cmp in lst_components:
        # get info from the component definition
        name: str = list(cmp.keys())[0]
        elem = list(cmp.values())[0]
        extras: ExtraParams = elem if elem is not None else {}

        # add the base module in "name" to the globals if it is not there
        _add_modules_to_globals([name])
        # create and add the component to the registry
        component_factory(name, eval(name, COMP_GLOBALS), extras)


def register_components_from_yml(file_name: str) -> None:
    """Register the components defined in the yml file.

    Args:
        file_name: name of the yml file containing the components. For details look at `definition.md`.

    """
    # read the yml file as a dict
    defs: ComponentDefinition = {}
    with open(file_name, "r") as stream:
        try:
            defs = yaml.safe_load(stream)
        except yaml.YAMLError:
            log.error("The list of component definitions for pytorch couldn't be loaded.", exc_info=True)

    # convert dict into a list
    lst: List[ComponentDefinition] = []
    for key, value in defs.items():
        lst.append({key: value})
    register_components(lst)


def register_component(cls) -> None:
    """Define a decorator to register a component.

    Args:
        cls: class to be registered.
        module: module where the class is defined.

    Returns:
        cls: class to be registered.

    Example:
        >>> @register_component
        >>> class YourComponent(Component):
        >>>     ...

        Will register the component as:
        limbus.component.XXX___YourComponent

        If you are in a notebook or script they will be registered as:
        limbus.component.limbus___YourComponent

    """
    module = cls.__module__
    if cls.__module__ == "limbus.components.base" or cls.__module__ == "__main__":
        # the base components will appear in the limbus module
        module = "limbus"
    _add_modules_to_globals([module])
    str_module: str = module.replace(".", "___")
    COMP_GLOBALS[f"{str_module}___{cls.__name__}"] = cls
    return cls
