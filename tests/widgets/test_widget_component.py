from limbus.widgets import WidgetState, WidgetComponent


class TestWidgetComponent:
    def test_smoke(self):
        cmp = WidgetComponent("yuhu")
        assert cmp.name == "yuhu"
        assert cmp.inputs is not None
        assert cmp.outputs is not None
        assert cmp.properties is not None
        assert cmp.widget_state == WidgetState.ENABLED

    def test_widget_state(self):
        cmp = WidgetComponent("yuhu")
        assert cmp.widget_state == WidgetState.ENABLED
        cmp.widget_state = WidgetState.DISABLED
        assert cmp.widget_state == WidgetState.DISABLED
