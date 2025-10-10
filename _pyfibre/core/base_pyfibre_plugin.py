from envisage.plugin import Plugin
from traits.api import Property, List, Str, Int

from pyfibre.ids import MULTI_IMAGE_FACTORIES, plugin_id


class BasePyFibrePlugin(Plugin):
    """Plugin that can be extended to provide additional multi
    image objects, such as IMultiImageFactory class"""

    id = Property(Str)

    #: Name of the plugin
    name = Property(Str)

    #: Version number of the plugin
    version = Property(Int)

    #: List of MultiImageFactories contributed by plugin
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

    def _get_id(self):
        """Getter that shadows public method for name"""
        return plugin_id(self.name, self.version)

    def _get_name(self):
        """Getter that shadows public method for name"""
        return self.get_name()

    def _get_version(self):
        """Getter that shadows public method for version"""
        return self.get_version()

    def get_name(self):
        """Returns name of plugin"""
        raise NotImplementedError

    def get_version(self):
        """Returns version number of plugin"""
        raise NotImplementedError

    def get_test_files(self):
        """Returns a list of image files to use for integration tests"""
        return []

    def get_multi_image_factories(self):
        """Returns a list of classes that provide an interface
        to IMultiImageFactory"""
        raise NotImplementedError
