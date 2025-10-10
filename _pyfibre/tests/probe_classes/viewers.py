from traits.api import Event

from pyfibre.core.base_multi_image_viewer import (
    BaseDisplayTab, BaseMultiImageViewer)


class ProbeDisplayTab(BaseDisplayTab):

    updated = Event()

    def update_tab(self):
        self.updated = True

    def customise_plot(self, plot):
        return plot


class ProbeMultiImageViewer(BaseMultiImageViewer):

    def create_display_tabs(self):
        return [ProbeDisplayTab()]

    def update_display_tabs(self):
        self.display_tabs[0].updated = True
