"""All the components will be added in this module automatically."""
from pathlib import Path

from limbus.core import factory
# this line needs to be at the top of the file
factory.COMP_GLOBALS = globals()
from limbus.core import Component  # noqa: E402
from limbus.core import register_components_from_yml  # noqa: E402
from limbus.components import base  # noqa: E402

# register all components in the default ymls
DEFAULT_COMPONENT_FILES = ["pytorch.yml", "kornia.yml"]
for comp_file in DEFAULT_COMPONENT_FILES:
    register_components_from_yml(str(Path(__file__).parent / comp_file))
