"""Module containing the basic functions to create components automatically."""
import inspect
from typing import Callable, Union, Dict, Any, cast, List, TypedDict, Optional, NamedTuple, Tuple
import importlib
from importlib.machinery import ModuleSpec
from importlib.abc import Loader
import logging
import types
from pathlib import Path
import sys

import yaml
import typeguard
import torch.fx
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
    idx: int  # signature idx if there are several signatures
    init: Dict[str, str]
    params: Dict[str, str]
    returns: Union[str, Dict[str, str], List[str]]


ComponentDefinition = Dict[str, ExtraParams]


def _add_modules_to_globals(modules: List[str], dynamic_module: str) -> None:
    """Add the modules in the list to the globals where the component will be defined.

    Args:
        modules: list of modules to add to the globals. The first module in each element will be added.
        dynamic_module: the module where the component will be defined. The entire list of modules will be added to the
            globals.

    """
    globals: Dict[str, Any] = COMP_GLOBALS
    if dynamic_module is not None:
        # create and add the dynamic module to the globals (it can require to create a tree)
        dyn_modules: List[str] = dynamic_module.split('.')
        for dyn_mod in dyn_modules:
            # check if the module exists and add it to the globals if it doesn't
            if dyn_mod not in globals:
                globals[dyn_mod] = types.ModuleType(f"{globals['__name__']}.{dyn_mod}")
                # add minimal required modules to the dynamic module
                globals[dyn_mod].Component = Component
            # set the new globals
            globals = globals[dyn_mod].__dict__

    # import the first module in each element of the list into the deepest module in the dynamic module
    for module in modules:
        if module.find(".") != -1:
            module = module.split(".")[0]
        if module not in globals:
            globals[module] = importlib.import_module(module)


def _get_annotation(annotation: Any, dynamic_module: str) -> str:
    """Get the string representation of the type of a parameter and add the module of the type to the globals."""
    # if it is a standard type...
    if typeguard.isclass(annotation):
        # add the module where the type is defined to ensure it is accesible.
        _add_modules_to_globals([annotation.__module__], dynamic_module)
        return f"{annotation.__module__}.{annotation.__name__}"
    # else we assume it is a typing expresion...
    else:
        # TODO: check if inside the typing expression the base types are always defined within module
        _add_modules_to_globals(["typing"], dynamic_module)
        return str(annotation)


def _get_params(sign: inspect.Signature, dynamic_module: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for param in sign.parameters.values():
        params[param.name] = _get_annotation(param.annotation, dynamic_module)
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


def component_factory(module: str, name: str, extra: ExtraParams) -> None:
    """Generate a Component class for a given callable/nn.Module and add it to the globals.

    Args:
        module: module where the wrapped component is going to be defined. E.g. aaa.bbb
        name: name of the function to be wrapped as a component. E.g. aaa.bbb.ccc
        extra: extra parameters used to create the component.

    """
    # search for the correct dynamic module
    globals = COMP_GLOBALS
    for mod in module.split('.'):
        globals = globals[mod].__dict__
    # get the original callable
    callable_to_wrap: Union[Callable, type] = eval(name, globals)

    # add the NoneType type to the globals.
    # NOTE: This type is returned by inspect.signature but it does not exist
    globals['NoneType'] = NoneType
    # add the typing module to the globals.
    _add_modules_to_globals(["typing"], module)

    # 1. define forward and signature vars
    # ------------------------------------
    # ATTENTION: In this function "callable_forward" var is used as pointer to change
    # the job done inside the new component class. The register_inputs() and register_outputs() methods use directly
    # this var to get the input and output params. The forward() is a bit trickier since the self._callable_forward
    # var is assigned inside the template of the component class.
    callable_forward: Union[Callable, type] = callable_to_wrap
    callable_signature: inspect.Signature

    is_torch: bool = (name[0: name.find(".")] == "torch")
    # We need to distinguish between functions, builtins and nn.Modules.
    if inspect.isfunction(callable_to_wrap):
        callable_signature = inspect.signature(callable_to_wrap)
    # torch functions are builtin and their signatures mast be imported in a special way
    elif is_torch and inspect.isbuiltin(callable_to_wrap):
        callable_forward = callable_to_wrap
        # builtsins in torch can be overloaded, so we need to select one of the signatures (by default the first one)
        signs: Optional[List[inspect.Signature]] = (
            torch.fx.operator_schemas.get_signature_for_torch_op(callable_to_wrap))
        assert signs is not None, f"Something weird happened with {name}. It should have a signature."
        sign_idx: int = extra.get("idx", 0)
        callable_signature = signs[sign_idx]
    elif isinstance(callable_to_wrap, type) and nn.Module in inspect.getmro(callable_to_wrap):
        # in this case "callable_forward" refers to the forward method in the original nn.Module
        callable_forward = callable_to_wrap.forward  # type: ignore  # mypy don't know about the forward method
        callable_signature = inspect.signature(callable_forward)
    else:
        raise TypeError(f"{callable_to_wrap} must be a function, a torch builtin or an nn.Module.")

    # 2. define the forward(), register_inputs() and register_outputs() methods
    # -------------------------------------------------------------------------
    def forward(self) -> ComponentState:  # noqa: D417
        """Run the component."""
        args: Dict[str, Any] = {}
        for param in self._inputs.get_params():
            args[param] = self._inputs.get_param(param)
        # mypy cannot infer that the class has this method
        res = self._callable(**args)  # type: ignore
        if len(self._outputs.get_params()) > 1:
            for idx, param in enumerate(self._outputs.get_params()):
                self._outputs.set_param(param, res[idx])
        else:
            param = list(self._outputs.get_params())[0]
            self._outputs.set_param(param, res)
        return ComponentState.OK

    def _import_if_it_is_possible(modules: List[str], dynamic_module: str) -> None:
        # if there is a module to import, let's import it
        try:
            _add_modules_to_globals(modules, dynamic_module)
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
            # NOTE: callable_signature needs to be predefined
            # if there are no params in extra, we use the signature of the function
            params = list(callable_signature.parameters.values())
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
                _import_if_it_is_possible([p_annotation], module)
                # convert to the proper type the type and default value
                p_default = eval(p_default, globals) if not p_empty else None
                p_annotation = eval(p_annotation, globals)
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
        # if the return_annotation is None, we need to convert it into its type (anyway this shouldn't happen)
        if return_annotation is None:
            return_annotation = NoneType
        if (not typeguard.isclass(return_annotation) and return_annotation._name == "Tuple"
                and len(return_annotation.__args__) > 1 and Ellipsis not in return_annotation.__args__):
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
            # NOTE: callable_signature needs to be predefined
            # if there are no returns in extra, we use the signature to get them
            return_annotation: Any = callable_signature.return_annotation
        # get returns only using the signature
        if not returns:
            if isinstance_namedtuple(return_annotation):
                # variable number of outputs
                for k, v in return_annotation._field_defaults.items():
                    outputs.declare(k, v)
            else:
                _helper_to_add_returns(outputs, return_annotation)

        #############################################################################################
        # Valid formats for the "returns" variable:
        #    - "" if we want the default return of the callable
        #    - "type" to force the return type of the callable
        #    - List[str] to force the name of the output parameter of the callable
        #    - Dict[str, str] to force the name and the type of the output parameter of the callable
        #############################################################################################
        elif isinstance(returns, str):
            _import_if_it_is_possible([returns], module)
            outputs.declare("out", eval(returns, globals))
        elif isinstance(returns, list):
            _helper_to_add_returns(outputs, return_annotation, returns)
        elif isinstance(returns, dict):
            ret_name: List[str] = []
            return_annotation = ""
            for key in returns:
                ret_name.append(key)
                _import_if_it_is_possible([returns[key]], module)
                return_annotation += f"{returns[key]}, "
            return_annotation = return_annotation[:-2]
            if len(ret_name) > 1:
                return_annotation = f"typing.Tuple[{return_annotation}]"
            _helper_to_add_returns(outputs, eval(return_annotation, globals), ret_name)
        else:
            raise TypeError("Invalid type definition for the output pins.")
        return outputs

    # 3. create the component class
    # -----------------------------
    # define the name of the component class by removing the first module name (remember that it is the one dynamically
    # created to contain the dynamic code)
    str_name = name[name.rfind('.') + 1:]
    # create the class
    if inspect.isfunction(callable_to_wrap) or inspect.isbuiltin(callable_to_wrap):
        # 1. define the class template
        func = (f"class {str_name}(Component):\n"
                f"    callable_object = callable_forward\n"
                f"    def __init__(self, name: str) -> None:\n"
                f"        super().__init__(name)\n"
                f"        self._callable = callable_forward\n")
        # 2. compile and change the keyword "callable_forward" for the real function to be run
        code = compile(func, f"dynamic module: {module}", "exec")
        eval(code, {"callable_forward": callable_forward}, globals)
        cls_prototype: str = f"{str_name}(name: str)"
    else:
        # In the case of an nn.Module we need to dinamically assign the params to the __init__ method and
        # create the original object.
        # 1. Write the parameters of the __init__ method and the call to the forward method
        init_params: Dict[str, str] = extra.get("init", {})
        if not init_params:
            init_params = _get_params(inspect.signature(callable_to_wrap), module)
        else:
            for param in init_params:
                _import_if_it_is_possible([init_params[param]], module)
        str_params_def, str_params = _get_params_as_def(init_params)

        # 2. write the template for the component
        str_params_def = f"self, name: str, {str_params_def}"  # add the name parameter required by the component
        func = (f"class {str_name}(Component):\n"
                f"    callable_object = {name}\n"
                f"    def __init__({str_params_def}) -> None:\n"
                f"        super().__init__(name)\n"
                f"        self._real_obj =  {name}({str_params})\n"
                f"        self._callable = self._real_obj.forward\n")
        # 3. compile the code and add the methods to the component class
        code = compile(func, f"dynamic module: {module}", "exec")
        eval(code, globals)
        cls_prototype = f"{str_name}({str_params_def})"

    # 4. create the component documentation
    # -------------------------------------
    def _gen_doc_params(params: Params) -> str:
        doc: str = ""
        for param in params:
            default: str = ""
            if not isinstance(param.value, NoValue):
                default = f" - Default: {param.value}"
            if inspect.isclass(param.type):
                doc += f'\t{param.name} ({param.type.__name__}){default}\n'
            else:
                doc += f'\t{param.name} ({param.type}){default}\n'
        return doc

    def autogenerate_documentation(cls_prototype: str, component: Component, callable_to_wrap: Callable) -> str:
        doc: str = f"Autogenerated component:\n\n\t{cls_prototype}\n\nArgs:\n\tname (str): name of the component\n"

        inputs: Params = component.register_inputs()
        if len(inputs) > 0:
            doc += f"\nInput params:\n"
            doc += _gen_doc_params(inputs)
        outputs: Params = component.register_outputs()
        if len(outputs) > 0:
            doc += f"\nOutput params:\n"
            doc += _gen_doc_params(outputs)
        doc += f"\nOriginal documentation:\n\n{inspect.getdoc(callable_to_wrap)}"
        return doc

    # set the documentation of the original class or function
    globals[str_name].forward = forward
    globals[str_name].register_inputs = register_inputs
    globals[str_name].register_outputs = register_outputs
    globals[str_name].__doc__ = autogenerate_documentation(cls_prototype, globals[str_name], callable_to_wrap)


def register_components(lst_components: List[ComponentDefinition]) -> None:
    """Create and register all the components of a list of components.

    Args:
        lst_components: List of components to register.

    """
    cmp: ComponentDefinition
    for cmp in lst_components:
        # get info from the component definition
        name: str = list(cmp.keys())[0]
        elem = list(cmp.values())[0]
        extras: ExtraParams = elem if elem is not None else {}
        if name.find(".") == -1:
            raise ValueError(f"The component name ({name}) must be in the format module.component")
        module: str = name[:name.rfind(".")]
        # add the base module in "name" to the globals if it is not there
        _add_modules_to_globals([name], module)
        # create and add the component to the registry
        component_factory(module, name, extras)


def register_components_from_yml(file_name: str) -> None:
    """Create and register components defined in the yml file.

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


def register_component(cls: Component, dst_module: str) -> None:
    """Register a already created component in a concrete module inside limbus.

    Args:
        cls: class to be registered.
        dst_module: module where the component will be defined.

    Example:
        >>> class YourComponent(Component):
        >>>     ...
        >>> register_component(YourComponent, "my_module0.my_module1")

        Will register the component as:
        limbus.component.my_module0.my_module1.YourComponent

    """
    _add_modules_to_globals([], dst_module)  # creates the dynamic module if it doesn't exist
    globals = COMP_GLOBALS
    for mod in dst_module.split("."):
        globals = globals[mod].__dict__

    # TODO: validate that this code covers all the cases
    module = cls.__module__
    if module != "__main__":
        # if the component belong to a module...
        _add_modules_to_globals([module], dst_module)
    globals[cls.__name__] = cls  # type: ignore  # mypy does not recognise __name__ as str


def register_components_from_module(file_name: str) -> None:
    """Register already existing components defined in a python module.

    NOTE: the destination module will be the filename without the extension.
          i.e. limbus.components.{filename}

    Args:
        file_name: name of the python module containing the components.

    """
    dst_module: str = Path(file_name).stem
    # create the module path inside "limbus.components"
    spec: Optional[ModuleSpec] = importlib.util.spec_from_file_location(f"limbus.components.{dst_module}", file_name)
    if spec is None or spec.loader is None:
        raise ValueError(f"The module {file_name} could not be imported.")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(module)
    # get access to all the components in the module
    globals = COMP_GLOBALS
    if module not in globals:
        globals[dst_module] = module
    else:
        raise ValueError(f"Module {dst_module} already exists in 'limbus.components'.")


def register_components_from_path(file_name: str) -> None:
    """Register components from yml or modules.

    This is a high level interface to register components that internally calls
        register_components_from_module() or register_components_from_yml()

    NOTE: from a module the destination module will be the filename without the extension.
        i.e. limbus.components.{filename}

    Args:
        file_name: name of the python module or yml file containing the components.

    """
    if Path(file_name).suffix == ".yml":
        register_components_from_yml(file_name)
    elif Path(file_name).suffix == ".py":
        register_components_from_module(file_name)
    else:
        raise ValueError(f"The file '{file_name}' must be a .yml or .py file.")


def deregister_all_components() -> None:
    """Remove all the registered component from the registry."""
    default_modules: List[str] = [cmp.split(".")[0] for cmp in COMP_GLOBALS["DEFAULT_COMPONENT_FILES"]]

    modules_with_components: List[str] = []
    for k, v in COMP_GLOBALS.items():
        if inspect.ismodule(v) and k not in COMP_GLOBALS["IMPORTED_MODULES"] and k not in default_modules:
            modules_with_components.append(k)

    for k in modules_with_components:
        COMP_GLOBALS.pop(k)
