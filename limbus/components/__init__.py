"""All the components will be added in this module automatically."""
from limbus.core import factory
# this line needs to be at the top of the file
factory.COMP_GLOBALS = globals()
from limbus.core import Component  # noqa: E402
from limbus.components import base  # noqa: E402
