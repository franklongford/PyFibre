import logging
import os

from pyface.tasks.api import TraitsTaskPane
from traits.api import (
    Instance, List, Property, Str, Enum, on_trait_change
)
from traitsui.api import (
    View, Group, Item, ListEditor, EnumEditor
)

from pyfibre.gui.image_tab import (
    ImageTab, TensorImageTab, NetworkImageTab, MetricTab)
from pyfibre.gui.segment_image_tab import SegmentImageTab
from pyfibre.io.object_io import (
    load_fibre_segments, load_cell_segments,
    load_fibre_networks, load_fibres)
from pyfibre.io.shg_pl_reader import SHGPLTransReader
from pyfibre.gui.file_display_pane import TableRow
from pyfibre.model.multi_image.base_multi_image import BaseMultiImage
from pyfibre.model.iterator import assign_images
from pyfibre.utilities import flatten_list

logger = logging.getLogger(__name__)


class ViewerPane(TraitsTaskPane):

    id = 'pyfibre.viewer_pane'

    name = 'Viewer Pane'

    image_tab_list = List(Instance(ImageTab))

    multi_image_tab = Instance(ImageTab)

    tensor_tab = Instance(ImageTab)

    network_tab = Instance(NetworkImageTab)

    fibre_tab = Instance(NetworkImageTab)

    fibre_segment_tab = Instance(SegmentImageTab)

    cell_segment_tab = Instance(SegmentImageTab)

    metric_tab = Instance(MetricTab)

    multi_image_reader = Instance(SHGPLTransReader)

    selected_row = Instance(TableRow)

    selected_image = Instance(BaseMultiImage)

    selected_label = Enum(values='image_labels')

    image_labels = Property(
        List(Str), depends_on='selected_image.image_dict'
    )

    list_editor = ListEditor(
        page_name='.label',
        use_notebook=True,
        dock_style='tab',
        style='custom'
    )

    traits_view = View(
        Group(
            Item('image_tab_list',
                 editor=list_editor,
                 style='custom'),
            Item('selected_label',
                 editor=EnumEditor(
                     name='object.image_labels'),
                 style='simple'),
            show_labels=False
        )
    )

    def _multi_image_reader_default(self):
        return SHGPLTransReader()

    def _get_image_labels(self):
        if self.selected_image is not None:
            return list(self.selected_image.image_dict.keys())
        return []

    @on_trait_change('selected_row')
    def open_file(self):
        """Opens corresponding to the first item in
        selected_rows"""

        filenames = assign_images(
            self.selected_row._dictionary)

        self.selected_image = self.multi_image_reader.load_multi_image(
            filenames
        )

    def _image_tab_list_default(self):

        return [self.multi_image_tab,
                self.tensor_tab,
                self.network_tab,
                self.fibre_tab,
                self.fibre_segment_tab,
                self.cell_segment_tab]

    def _multi_image_tab_default(self):
        return ImageTab(
            multi_image=self.selected_image,
            label='Loaded Image')

    def _tensor_tab_default(self):
        return TensorImageTab(
            multi_image=self.selected_image,
            label='Tensor Image')

    def _network_tab_default(self):
        return NetworkImageTab(
            multi_image=self.selected_image,
            label='Network Image')

    def _fibre_tab_default(self):
        return NetworkImageTab(
            multi_image=self.selected_image,
            label='Fibres', c_mode=1)

    def _fibre_segment_tab_default(self):
        return SegmentImageTab(
            multi_image=self.selected_image,
            label='Fibre Segments')

    def _cell_segment_tab_default(self):
        return SegmentImageTab(
            multi_image=self.selected_image,
            label='Cell Segments')

    def _update_shg_image_tabs(self, filename, image_name):

        networks = []
        fibres = []
        fibre_segments = []

        try:
            fibre_networks = load_fibre_networks(filename)
        except (IOError, EOFError):
            logger.info(
                f"Unable to display network for {image_name}")
        else:
            networks = [
                fibre_network.graph
                for fibre_network in fibre_networks]

            try:
                fibres = load_fibres(
                    filename, image=self.selected_image.shg_image)
            except (IOError, EOFError):
                fibres = flatten_list([
                    fibre_network.fibres
                    for fibre_network in fibre_networks])
            fibres = [fibre.graph for fibre in fibres]

        try:
            fibre_segments = load_fibre_segments(
                filename, image=self.selected_image.shg_image)
        except (AttributeError, IOError, EOFError):
            logger.debug(
                f"Unable to display fibre segments for {image_name}")

        self.network_tab.networks = networks
        self.fibre_tab.networks = fibres
        self.fibre_segment_tab.segments = fibre_segments

    def _update_pl_image_tabs(self, filename, image_name):

        cell_segments = []

        try:
            cell_segments = load_cell_segments(
                filename, image=self.selected_image.pl_image)
        except (AttributeError, IOError, EOFError):
            logger.debug(
                f"Unable to display cell segments for {image_name}")

        self.cell_segment_tab.segments = cell_segments

    @on_trait_change('selected_image')
    def update_viewer(self):

        if self.selected_row is not None:
            image_name = os.path.basename(self.selected_row.name)
            image_path = os.path.dirname(self.selected_row.name)
            data_dir = f"{image_path}/{image_name}-pyfibre-analysis/data/"
            filename = data_dir + image_name

            for image_tab in self.image_tab_list:
                image_tab.multi_image = self.selected_image

            self._update_shg_image_tabs(filename, image_name)
            self._update_pl_image_tabs(filename, image_name)

    @on_trait_change('selected_label')
    def update_label(self):
        for image_tab in self.image_tab_list:
            image_tab.selected_label = self.selected_label
