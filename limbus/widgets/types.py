"""Module containing the visualization interfaces."""
from abc import abstractmethod
import math
from typing import Optional, Union, List, Callable, Set, Tuple, Dict, Any
import logging
import functools

try:
    # NOTE: we import the cv2 & visdom modules here to avoid having it as a dependency
    # for the whole project.
    import cv2
    import visdom
except ImportError:
    pass

import torch
import kornia
import numpy as np

from limbus.core import Component
from limbus import widgets

log = logging.getLogger(__name__)


def _get_component_from_args(*args, **kwargs) -> Component:
    # NOTE: this is a hack to get the component from the args. We know that the first argument is the component in all
    # the methods.
    if len(args) > 0:
        return args[0]
    elif "component" in kwargs:
        return kwargs["component"]
    else:
        raise ValueError("No component found in args or kwargs.")


def _get_title_from_args(*args, **kwargs) -> str:
    # NOTE: this is a hack to get the title from the args. We know that the second argument is the title in all the
    # methods.
    if len(args) > 1:
        return args[1]
    elif "title" in kwargs:
        return kwargs["title"]
    else:
        raise ValueError("No title found in args or kwargs.")


def _set_title_in_args(title: str, args: Tuple[Any, ...], kwargs: Dict[Any, Any]
                       ) -> Tuple[Tuple[Any, ...], Dict[Any, Any]]:
    # NOTE: this is a hack to update the title from the args. We know that the second argument is the title in all the
    # methods.
    if len(args) > 1:
        new_args: List[Any] = list(args)
        new_args[1] = title
        return (tuple(new_args), kwargs)
    elif "title" in kwargs:
        kwargs.update({"title": title})
    return (args, kwargs)


# This is a decorator that will disable the method if the visualization is not enabled.
def is_enabled(func: Callable) -> Callable:
    """Return None if viz is not enabled."""
    @functools.wraps(func)
    def wrapper_check_component_disabled(self, *args, **kwargs) -> Any:
        vz = widgets.get(False)
        if vz is None or not vz.enabled:
            return None
        if not vz.force_viz and _get_component_from_args(*args, **kwargs).widget_state != widgets.WidgetState.ENABLED:
            return None
        return func(self, *args, **kwargs)
    return wrapper_check_component_disabled


def set_title(func: Callable) -> Callable:
    """Set the title to be used in the viz if title param is empty."""
    @functools.wraps(func)
    def wrapper_set_title(self, *args, **kwargs) -> Any:
        comp_name: str = _get_component_from_args(*args, **kwargs).name
        title: str = _get_title_from_args(*args, **kwargs)
        if title == "":
            title = comp_name
            args, kwargs = _set_title_in_args(title, args, kwargs)
        return func(self, *args, **kwargs)
    return wrapper_set_title


class Viz:
    """Base class containing the method definitions for the visualization backends.

    IMPORTANT NOTE to create or add new visualization backends/methods:
        All the methods showing data should be decorated with @is_enabled and @set_title
        All the methods showing data must have as first argument the "component" and as second argument the "title".
            With those argument names since the decorators use them.

    """
    def __init__(self) -> None:
        self._enabled: bool = False
        # by default the components control if they will viz or not. However, in some cases with this parameter we can
        # force to viz even if the viz is not enabled.
        self._force_viz: bool = False

    @property
    def enabled(self) -> bool:
        """Whether the visualization is enabled."""
        return self._enabled

    @property
    def force_viz(self) -> bool:
        """Whether the visualization is forced for all the components."""
        return self._force_viz

    @force_viz.setter
    def force_viz(self, force_viz: bool) -> None:
        """Force viz for all the components."""
        self._force_viz = force_viz

    @abstractmethod
    def check_status(self) -> bool:
        """Check if the connection is alive and try to reconnect if connection is lost."""
        raise NotImplementedError

    @abstractmethod
    def show_image(self, component: Component, title: str, image: torch.Tensor):
        """Show an image.

        Args:
            component: component that calls this method.
            title: Title of the window.
            image: Tensor with shape ([1, 3] x H x W) or (H x W). Values can be float in [0, 1] or uint8 in [0, 255].

        """
        raise NotImplementedError

    @abstractmethod
    def show_images(self, component: Component, title: str,
                    images: Union[torch.Tensor, List[torch.Tensor]],
                    nrow: Optional[int] = None
                    ) -> None:
        """Show a batch of images.

        Args:
            component: component that calls this method.
            title: Title of the window.
            images: 4D Tensor with shape (B x [1, 3] x H x W) or a list of tensors with the same shape
                ([1, 3] x H x W) or (H x W).
            nrow (optional): Number of images in each row. Default: None -> sqrt(len(images)).

        """
        raise NotImplementedError

    @abstractmethod
    def show_text(self, component: Component, title: str, text: str, append: bool = False):
        """Show text.

        Args:
            component: component that calls this method.
            title: Title of the window.
            text: Text to be displayed.
            append (optional): If True, the text is appended to the previous text. Default: False.

        """
        raise NotImplementedError


class Visdom(Viz):
    """Visdom visualization backend."""
    VISDOM_PORT = 8097

    def __init__(self) -> None:
        super().__init__()
        try:
            import visdom
        except:
            raise ImportError("To use Visdom as backend install the widgets extras: "
                              "pip install limbus[widgets]")
        self._vis: Optional[visdom.Visdom] = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            self._vis = visdom.Visdom(port=Visdom.VISDOM_PORT, raise_exceptions=True)
            self._enabled = True
        except:
            self._enabled = False

        if not self._enabled:
            log.warning("Visualization is disabled!!!")
            return

        assert self._vis is not None, "Visdom is not initialized."
        if not self._vis.check_connection():
            self._enabled = False
            log.warning("Error connecting with the visdom server.")

    def check_status(self) -> bool:
        """Check if the connection is alive and try to reconnect if connection is lost."""
        if self._vis is None:
            self._try_init()
        else:
            self._enabled = self._vis.check_connection()
        return self._enabled

    @is_enabled
    @set_title
    def show_image(self, component: Component, title: str, image: torch.Tensor) -> None:
        """Show an image.

        Args:
            component: component that calls this method.
            title: Title of the window.
            image: Tensor with shape ([1, 3] x H x W) or (H x W). Values can be float in [0, 1] or uint8 in [0, 255].

        """
        opts = {"title": title}
        assert self._vis is not None, "Visdom is not initialized."
        self._vis.image(image, win=title, opts=opts)

    @is_enabled
    @set_title
    def show_images(self, component: Component, title: str,
                    images: Union[torch.Tensor, List[torch.Tensor]],
                    nrow: Optional[int] = None
                    ) -> None:
        """Show a batch of images.

        Args:
            component: component that calls this method.
            title: Title of the window.
            images: 4D Tensor with shape (B x [1, 3] x H x W) or a list of tensors with the same shape
                ([1, 3] x H x W) or (H x W).
            nrow (optional): Number of images in each row. Default: None -> sqrt(len(images)).

        """
        opts = {"title": title}
        if nrow is None:
            l: int = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
            nrow = math.ceil(math.sqrt(l))
        assert self._vis is not None, "Visdom is not initialized."
        self._vis.images(images, win=title, opts=opts, nrow=nrow)

    @is_enabled
    @set_title
    def show_text(self, component: Component, title: str, text: str, append: bool = False):
        """Show text.

        Args:
            component: component that calls this method.
            title: Title of the window.
            text: Text to be displayed.
            append (optional): If True, the text is appended to the previous text. Default: False.

        """
        assert self._vis is not None, "Visdom is not initialized."
        opts = {"title": title}
        self._vis.text(text, win=title, append=append, opts=opts)


class Console(Viz):
    """COnsole visualization backend."""

    def __init__(self) -> None:
        super().__init__()
        self._enabled = True

    def check_status(self) -> bool:
        """Check if the connection is alive and try to reconnect if connection is lost."""
        return self._enabled

    @is_enabled
    @set_title
    def show_image(self, component: Component, title: str, image: torch.Tensor):
        """Show an image.

        Args:
            component: component that calls this method.
            title: Title of the window.
            image: Tensor with shape ([1, 3] x H x W) or (H x W). Values can be float in [0, 1] or uint8 in [0, 255].

        """
        log.warning("Console visualization does not show images.")

    @is_enabled
    @set_title
    def show_images(self, component: Component, title: str,
                    images: Union[torch.Tensor, List[torch.Tensor]],
                    nrow: Optional[int] = None
                    ) -> None:
        """Show a batch of images.

        Args:
            component: component that calls this method.
            title: Title of the window.
            images: 4D Tensor with shape (B x [1, 3] x H x W) or a list of tensors with the same shape
                ([1, 3] x H x W) or (H x W).
            nrow (optional): Number of images in each row. Default: None -> sqrt(len(images)).

        """
        log.warning("Console visualization does not show images.")

    @is_enabled
    @set_title
    def show_text(self, component: Component, title: str, text: str, append: bool = False):
        """Show text.

        Args:
            component: component that calls this method.
            title: Title of the window.
            text: Text to be displayed.
            append (optional): If True, the text is appended to the previous text. Default: False.

        """
        log.info(f" {title}: {text}")


class OpenCV(Console):
    """Console visualization backend + openCV for images."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import cv2
        except:
            raise ImportError("To use OpenCV as backend install the widgets extras: "
                              "pip install limbus[widgets]")

    @is_enabled
    @set_title
    def show_image(self, component: Component, title: str, image: torch.Tensor):
        """Show an image.

        Args:
            component: component that calls this method.
            title: Title of the window.
            image: Tensor with shape ([1, 3] x H x W) or (H x W). Values can be float in [0, 1] or uint8 in [0, 255].

        """
        # inspired by the image() function in visdom.__init__.py.
        # convert image type to uint8 [0, 255]
        if image.dtype in [torch.float, torch.float32, torch.float64]:
            if image.max() <= 1:
                image = image * 255.0
            image = image.byte()

        if image.ndim == 3 and image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # convert image shape to 3xHxW
        if image.ndim == 2:
            image = image.unsqueeze(0)
            image = image.repeat(3, 1, 1)

        np_img: np.ndarray = kornia.tensor_to_image(image)
        cv2.imshow(title, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    @is_enabled
    @set_title
    def show_images(self, component: Component, title: str,
                    images: Union[torch.Tensor, List[torch.Tensor]],
                    nrow: Optional[int] = None
                    ) -> None:
        """Show a batch of images.

        Args:
            component: component that calls this method.
            title: Title of the window.
            images: 4D Tensor with shape (B x [1, 3] x H x W) or a list of tensors with the same shape
                ([1, 3] x H x W) or (H x W).
            nrow (optional): Number of images in each row. Default: None -> sqrt(len(images)).

        """
        # inspired by the images() function in visdom.__init__.py.
        if isinstance(images, list):
            # NOTE that stack adds a dim to the result even if there was a single image.
            # so, if images is [(Hx W)] then the stack shape is (1 x H x W)
            images = torch.stack(images, 0)
            if images.ndim == 3:
                # if images are (H x W) convert tensor to (B x 1 x H x W)
                images = images.unsqueeze(1)
        else:
            # in this case we will assume a single image was passed.
            if images.ndim == 3:
                images = images.unsqueeze(0)

        # convert tensor shape from (B x C x H x W) to a grid with shape (C x H' x W')
        if nrow is None:
            nrow = math.ceil(math.sqrt(images.shape[0]))
        elif nrow > images.shape[0]:
            nrow = images.shape[0]
        ncol: int = math.ceil(images.shape[0] / nrow)
        padding: int = 4
        grid: torch.Tensor = torch.zeros((images.shape[1],
                                          ncol * (images.shape[2] + padding) - padding,
                                          nrow * (images.shape[3] + padding) - padding)).to(images)
        j: int = 0
        h: int = images.shape[2] + padding
        w: int = images.shape[3] + padding
        for idx in range(images.shape[0]):
            i = idx % nrow
            j = idx // nrow
            grid[:, j * h:j * h + images.shape[2], i * w:i * w + images.shape[3]] = images[idx]
        self.show_image(component, title, grid)
