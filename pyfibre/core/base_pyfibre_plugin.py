from envisage.plugin import Plugin
from traits.trait_types import List

from pyfibre.ids import MULTI_IMAGE_FACTORIES


class BasePyFibrePlugin(Plugin):
    """Plugin that can be extended to provide additional multi
    image objects, such as IMultiImageFactory class"""

    multi_image_factories = List(
        contributes_to=MULTI_IMAGE_FACTORIES
    )

    def __init__(self, *args, **kwargs):

        multi_image_factories = [
            cls() for cls in
            self.get_multi_image_factories()
        ]

        super(BasePyFibrePlugin, self).__init__(
            *args,
            multi_image_factories=multi_image_factories,
            ** kwargs
        )

    def get_multi_image_factories(self):
        """Returns a list of classes that provide an interface
        to IMultiImageFactory"""
        raise NotImplementedError
