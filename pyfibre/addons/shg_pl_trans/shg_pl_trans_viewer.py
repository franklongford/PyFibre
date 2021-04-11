import logging
import os

from traits.api import Instance, on_trait_change

from pyfibre.core.base_multi_image_viewer import BaseMultiImageViewer
from pyfibre.gui.image_tab import ImageTab, NetworkImageTab, TensorImageTab
from pyfibre.gui.metric_tab import ImageMetricTab
from pyfibre.gui.segment_image_tab import SegmentImageTab
from pyfibre.io.object_io import (
    load_fibre_segments, load_cell_segments,
    load_fibre_networks, load_fibres)
from pyfibre.io.database_io import (
    load_database
)
from pyfibre.utilities import flatten_list


logger = logging.getLogger(__name__)


class SHGPLTransViewer(BaseMultiImageViewer):

    multi_image_tab = Instance(ImageTab)

    tensor_tab = Instance(ImageTab)

    network_tab = Instance(NetworkImageTab)

    fibre_tab = Instance(NetworkImageTab)

    fibre_segment_tab = Instance(SegmentImageTab)

    cell_segment_tab = Instance(SegmentImageTab)

    def _multi_image_tab_default(self):
        return ImageTab(
            multi_image=self.multi_image,
            label='Loaded Image')

    def _tensor_tab_default(self):
        return TensorImageTab(
            multi_image=self.multi_image,
            label='Tensor Image')

    def _network_tab_default(self):
        return NetworkImageTab(
            multi_image=self.multi_image,
            label='Network Image')

    def _fibre_tab_default(self):
        return NetworkImageTab(
            multi_image=self.multi_image,
            label='Fibres', c_mode=1)

    def _fibre_segment_tab_default(self):
        return SegmentImageTab(
            multi_image=self.multi_image,
            label='Fibre Segments')

    def _cell_segment_tab_default(self):
        return SegmentImageTab(
            multi_image=self.multi_image,
            label='Cell Segments')

    def _update_shg_image_tabs(self, filename, image_name):

        networks = []
        fibres = []
        fibre_segments = []

        try:
            fibre_networks = load_fibre_networks(filename)
        except (IOError, EOFError):
            logger.debug(
                f"Unable to display network for {image_name}")
        else:
            networks = [
                fibre_network.graph
                for fibre_network in fibre_networks]

            try:
                fibres = load_fibres(filename)
            except (IOError, EOFError):
                fibres = flatten_list([
                    fibre_network.fibres
                    for fibre_network in fibre_networks])
            fibres = [fibre.graph for fibre in fibres]

        try:
            fibre_segments = load_fibre_segments(
                filename, intensity_image=self.multi_image.shg_image)
        except (AttributeError, IOError, EOFError):
            logger.debug(
                f"Unable to display fibre segments for {image_name}")

        self.network_tab.networks = networks
        self.fibre_tab.networks = fibres
        self.fibre_segment_tab.segments = fibre_segments

        try:
            self.fibre_segment_tab.data = load_database(
                filename, file_type='fibre_metric'
            )
        except (AttributeError, IOError, EOFError):
            logger.debug(
                f"Unable to load fibre metrics for {image_name}")

    def _update_pl_image_tabs(self, filename, image_name):

        cell_segments = []

        try:
            cell_segments = load_cell_segments(
                filename, intensity_image=self.multi_image.pl_image)
        except (AttributeError, IOError, EOFError):
            logger.debug(
                f"Unable to display cell segments for {image_name}")

        self.cell_segment_tab.segments = cell_segments

        try:
            self.cell_segment_tab.data = load_database(
                filename, file_type='cell_metric'
            )
        except (AttributeError, IOError, EOFError):
            logger.debug(
                f"Unable to load cell metrics for {image_name}")

    @on_trait_change('selected_tab')
    def update_tab(self, object, name, old, new):
        self.selected_tab.multi_image = self.multi_image
        if old.selected_label is not None:
            self.selected_tab.selected_label = old.selected_label

    def create_display_tabs(self):

        return [self.multi_image_tab,
                self.tensor_tab,
                self.network_tab,
                self.fibre_tab,
                self.fibre_segment_tab,
                self.cell_segment_tab]

    def update_display_tabs(self):

        image_name = self.multi_image.name
        image_path = self.multi_image.path

        data_path = os.path.join(
            image_path, f"{image_name}-pyfibre-analysis", "data")
        filename = os.path.join(data_path, image_name)

        self._update_shg_image_tabs(filename, image_name)
        self._update_pl_image_tabs(filename, image_name)
        self.selected_tab.multi_image = self.multi_image
