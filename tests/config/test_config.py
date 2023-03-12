"""Config tests."""
import sys

from torch import nn


def remove_limbus_imports():
    """Remove limbus dependencies from sys.modules."""
    for key in list(sys.modules.keys()):
        if key.startswith("limbus"):
            del sys.modules[key]


def test_torch_base_class():
    remove_limbus_imports()
    from limbus_config import config
    config.COMPONENT_TYPE = "torch"
    import limbus
    mro = limbus.Component.__mro__
    remove_limbus_imports()
    assert len(mro) == 3
    assert nn.Module in mro


def test_generic_base_class():
    remove_limbus_imports()
    import limbus
    mro = limbus.Component.__mro__
    remove_limbus_imports()
    assert len(mro) == 2
    assert nn.Module not in mro
