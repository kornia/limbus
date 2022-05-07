"""All the components will be added in this module automatically."""
from pathlib import Path

from limbus.core import factory
# this line needs to be at the top of the file. Otherwise, the components will not be registered.
factory.COMP_GLOBALS = globals()

from limbus.core import Component  # noqa: E402
from limbus.core import register_components_from_yml, register_components_from_module, register_component  # noqa: E402

# IMPORTANT NOTE: In order to be able to deregister components with safety we need to list all the imported modules.
IMPORTED_MODULES = ["factory"]

# register all the default components (the name of the file must be the name of the module)
# check the deregister_all_components function to see how this var is used.
DEFAULT_COMPONENT_FILES = ["torch.yml", "kornia.yml", "base.py"]
for comp_file in DEFAULT_COMPONENT_FILES:
    path = Path(__file__).parent / comp_file
    if path.suffix == ".py":
        register_components_from_module(str(path))
    else:
        register_components_from_yml(str(path))
