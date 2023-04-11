"""Example with an app managing the pipeline."""
import asyncio

try:
    import aioconsole
except:
    raise ImportError("aioconsole is required to run this script. Install it with: pip install aioconsole")

from limbus.core import VerboseMode, App, async_utils
import limbus.widgets
try:
    import limbus_components as components
except ImportError:
    raise ImportError("limbus-components is required to run this script."
                      "Install the package with: "
                      "'pip install limbus-components@git+https://git@github.com/kornia/limbus-components.git'")

# Init the widgets backend
limbus.widgets.set_type("OpenCV")


class WebcamApp(App):
    """Example with an app managing a pipeline using a webcam."""
    def create_components(self):  # noqa: D102
        self._webcam = components.base.Webcam(name="webcam", batch_size=1)  # type: ignore
        self._show = components.base.ImageShow(name="show")  # type: ignore
        self._accum = components.base.Accumulator(name="acc", elements=2)  # type: ignore
        self._cat = components.torch.Cat(name="stack")  # type: ignore

    def connect_components(self):  # noqa: D102
        self._webcam.outputs.image >> self._accum.inputs.inp
        self._accum.outputs.out >> self._cat.inputs.tensors
        self._cat.outputs.out >> self._show.inputs.image

    def run(self, iters: int = 0):  # noqa: D102
        self._pipeline.set_verbose_mode(VerboseMode.PIPELINE)
        # self.pipeline.run(iters)
        async_utils.run_coroutine(self._app(self._pipeline))

    @staticmethod
    def _print_help() -> None:
        """Print the help message."""
        print(
            '\n\nOPTIONS MENU:\n'
            'Press "o" to run one pipeline iteration.\n'
            'Press "f" to run the pipeline forever.\n'
            'Press "r" to resume the pipeline.\n'
            'Press "p" to pause the pipeline.\n'
            'Press "vc" COMPONENT verbose state.\n'
            'Press "vp" PIPELINE verbose state.\n'
            'Press "vd" DISABLED verbose state.\n'
            'Press "q" to stop and quit.')

    async def _app(self, pipeline) -> None:
        """Run the interface."""
        while True:
            self._print_help()
            key_in = await aioconsole.ainput('Option:')
            if key_in == 'o':
                asyncio.create_task(pipeline.async_run(1))
            elif key_in == 'f':
                asyncio.create_task(pipeline.async_run())
            elif key_in == 'r':
                pipeline.resume()
            elif key_in == 'p':
                pipeline.pause()
            elif key_in == 'vc':
                pipeline.set_verbose_mode(VerboseMode.COMPONENT)
            elif key_in == 'vp':
                pipeline.set_verbose_mode(VerboseMode.PIPELINE)
            elif key_in == 'vd':
                pipeline.set_verbose_mode(VerboseMode.DISABLED)
            elif key_in == 'q':
                pipeline.stop()
                break


WebcamApp().run()
