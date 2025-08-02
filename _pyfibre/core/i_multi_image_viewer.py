from chaco.api import Plot
from traits.api import Interface, Instance, List, Str

from .base_multi_image import BaseMultiImage


class IDisplayTab(Interface):

    label = Str()

    plot = Instance(Plot)


class IMultiImageViewer(Interface):

    multi_image = Instance(BaseMultiImage)

    display_tabs = List(IDisplayTab)

    def create_display_tabs(self):
        """Returns a list of objects providing the IDisplayTab
        interface
        """

    def update_viewer(self, multi_image):
        """Returns a list of objects providing the IDisplayTab
        interface
        """
