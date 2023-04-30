import pytest

from limbus.core import App
from limbus_components.base import Constant, Printer


@pytest.mark.usefixtures("event_loop_instance")
class TestApp:
    def test_app(self):
        class MyApp(App):
            def create_components(self):  # noqa: D102
                self._constant = Constant("constant", "xyz")  # type: ignore
                self._print = Printer("print")  # type: ignore

            def connect_components(self):  # noqa: D102
                self._constant.outputs.out >> self._print.inputs.inp

        app = MyApp()
        app.run(1)
