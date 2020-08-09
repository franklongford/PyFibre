from envisage.api import ExtensionPoint
from envisage.core_plugin import CorePlugin
from traits.trait_types import List

from pyfibre.ids import MULTI_IMAGE_FACTORIES

from .i_multi_image_factory import IMultiImageFactory


class CorePyFibrePlugin(CorePlugin):
    """Inherits from the Envisage CorePlugin to include
    extra extension points for classes that fulfil the
    IMultiImageFactory interface"""

    id = 'pyfibre.core.pyfibre_plugin'

    #: Extension points for IMultiImageFactory
    multi_image_factories = ExtensionPoint(
        List(IMultiImageFactory),
        id=MULTI_IMAGE_FACTORIES
    )
