import logging

from pyface.tasks.api import TraitsTaskPane
from traits.api import (
    Instance, Dict, Type, Property
)
from traitsui.api import View, UItem

from pyfibre.gui.image_tab import ImageTab
from pyfibre.core.base_multi_image import BaseMultiImage
from pyfibre.core.base_multi_image_viewer import (
    BaseDisplayTab, BaseMultiImageViewer)

logger = logging.getLogger(__name__)


class BasicViewer(BaseMultiImageViewer):

    def create_display_tabs(self):
        """Returns a list of objects providing the IDisplayTab
        interface
        """
        return [ImageTab()]

    def update_display_tabs(self):
        """Updates each display tab when called"""
        self.display_tabs[0].multi_image = self.multi_image


class ViewerPane(TraitsTaskPane):

    id = 'pyfibre.viewer_pane'

    name = 'Viewer Pane'

    selected_image = Instance(BaseMultiImage)

    selected_viewer = Property(
        Instance(BaseDisplayTab), depends_on='selected_image')

    supported_viewers = Dict(Type, BaseMultiImageViewer)

    _basic_viewer = Instance(BaseMultiImageViewer)

    traits_view = View(
        UItem("selected_viewer", style="custom")
    )

    def __basic_viewer_default(self):
        return BasicViewer()

    def _selected_viewer_default(self):
        return self._basic_viewer

    def _get_selected_viewer(self):
        key = type(self.selected_image)
        if key in self.supported_viewers:
            viewer = self.supported_viewers[key]
        else:
            viewer = self._basic_viewer
        viewer.update_viewer(self.selected_image)
        return viewer
