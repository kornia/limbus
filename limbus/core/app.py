"""High level template to create apps."""
from abc import abstractmethod

from limbus.core import Pipeline, VerboseMode, Component


class App:
    """High level template to create an app."""
    def __init__(self):
        self.create_components()
        self.connect_components()
        # Create the pipeline
        self._pipeline = Pipeline()
        self._pipeline.add_nodes(self._get_component_attrs())
        self._pipeline.set_verbose_mode(VerboseMode.DISABLED)

    def _get_component_attrs(self) -> list[Component]:
        """Get the component attribute by name."""
        return [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), Component)]

    @abstractmethod
    def create_components(self):
        """Create the components of the app."""
        pass

    @abstractmethod
    def connect_components(self):
        """Connect the components of the app."""
        pass

    def run(self, iters: int = 0):
        """Run the app.

        Args:
            iters (optional): number of iters to be run. By default (0) all of them are run.

        """
        self._pipeline.run(iters)
