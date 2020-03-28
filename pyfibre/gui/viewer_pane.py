import logging
import os

from pyface.tasks.api import TraitsTaskPane
from traits.api import (
    Instance, List, on_trait_change
)
from traitsui.api import (
    View, Group, Item, ListEditor
)

from pyfibre.gui.image_tab import (
    ImageTab, NetworkImageTab, SegmentImageTab, MetricTab)
from pyfibre.io.object_io import (
    load_fibre_segments, load_cell_segments,
    load_fibre_networks, load_fibres)
from pyfibre.model.multi_image.multi_image import (
    SHGImage, SHGPLImage, SHGPLTransImage)
from pyfibre.io.shg_pl_reader import SHGPLTransReader
from pyfibre.gui.file_display_pane import TableRow
from pyfibre.model.multi_image.multi_image import MultiImage
from pyfibre.model.tools.figures import (
    create_tensor_image
)
from pyfibre.utilities import flatten_list

logger = logging.getLogger(__name__)


class ViewerPane(TraitsTaskPane):

    id = 'pyfibre.viewer_pane'

    name = 'Viewer Pane'

    image_tab_list = List(Instance(ImageTab))

    shg_image_tab = Instance(ImageTab)

    pl_image_tab = Instance(ImageTab)

    trans_image_tab = Instance(ImageTab)

    tensor_tab = Instance(ImageTab)

    network_tab = Instance(NetworkImageTab)

    fibre_tab = Instance(NetworkImageTab)

    fibre_segment_tab = Instance(SegmentImageTab)

    cell_segment_tab = Instance(SegmentImageTab)

    metric_tab = Instance(MetricTab)

    multi_image_reader = Instance(SHGPLTransReader)

    selected_row = Instance(TableRow)

    selected_image = Instance(MultiImage)

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
            show_labels=False
        )
    )

    def _multi_image_reader_default(self):
        return SHGPLTransReader()

    @on_trait_change('selected_row')
    def open_file(self):
        """Opens corresponding to the first item in
        selected_rows"""

        self.multi_image_reader.assign_images(
            self.selected_row._dictionary)

        self.selected_image = self.multi_image_reader.load_multi_image()

    def _image_tab_list_default(self):

        return [self.shg_image_tab,
                self.pl_image_tab,
                self.trans_image_tab,
                self.tensor_tab,
                self.network_tab,
                self.fibre_tab,
                self.fibre_segment_tab,
                self.cell_segment_tab]

    def _shg_image_tab_default(self):
        image_tab = ImageTab(label='SHG Image')
        if self.selected_image is not None:
            image_tab.image = self.selected_image.shg_image
        return image_tab

    def _pl_image_tab_default(self):
        image_tab = ImageTab(label='PL Image')
        if self.selected_image is not None:
            image_tab.image = self.selected_image.pl_image
        return image_tab

    def _trans_image_tab_default(self):
        image_tab = ImageTab(label='Transmission Image')
        if self.selected_image is not None:
            image_tab.image = self.selected_image.trans_image
        return image_tab

    def _tensor_tab_default(self):
        image_tab = ImageTab(label='Tensor Image')
        if self.selected_image is not None:
            tensor_image = create_tensor_image(
                self.selected_image.shg_image
            ) * 255.999
            image_tab.image = tensor_image
        return image_tab

    def _network_tab_default(self):
        image_tab = NetworkImageTab(label='Network')
        return image_tab

    def _fibre_tab_default(self):
        image_tab = NetworkImageTab(label='Fibres', c_mode=1)
        return image_tab

    def _fibre_segment_tab_default(self):
        image_tab = SegmentImageTab(label='Fibre Segments')
        return image_tab

    def _cell_segment_tab_default(self):
        image_tab = SegmentImageTab(label='Cell Segments')
        return image_tab

    def _update_shg_image_tabs(self, filename, image_name):

        self.shg_image_tab.image = self.selected_image.shg_image
        tensor_image = create_tensor_image(
            self.selected_image.shg_image
        ) * 255.999
        self.tensor_tab.image = tensor_image.astype('uint8')

        try:
            fibre_networks = load_fibre_networks(filename)
        except (IOError, EOFError):
            logger.info(
                f"Unable to display network for {image_name}")
        else:
            networks = [
                fibre_network.graph
                for fibre_network in fibre_networks]

            self.network_tab.networks = networks
            self.network_tab.image = self.selected_image.shg_image

            try:
                fibres = load_fibres(
                    filename, image=self.selected_image.shg_image)
            except (IOError, EOFError):
                fibres = flatten_list([
                    fibre_network.fibres
                    for fibre_network in fibre_networks])

            networks = [fibre.graph for fibre in fibres]

            self.fibre_tab.networks = networks
            self.fibre_tab.image = self.selected_image.shg_image

        try:
            fibre_segments = load_fibre_segments(
                filename, image=self.selected_image.shg_image)
        except (IOError, EOFError):
            logger.info(
                f"Unable to display fibre segments for {image_name}")
        else:
            self.fibre_segment_tab.segments = fibre_segments
            self.fibre_segment_tab.image = self.selected_image.shg_image

    def _update_pl_image_tabs(self, filename, image_name):

        self.pl_image_tab.image = self.selected_image.pl_image

        try:
            cell_segments = load_cell_segments(
                filename, image=self.selected_image.pl_image)
        except (AttributeError, IOError, EOFError):
            logger.debug(
                f"Unable to display cell segments for {image_name}")
        else:
            self.cell_segment_tab.segments = cell_segments
            self.cell_segment_tab.image = self.selected_image.pl_image

        if isinstance(self.selected_image, SHGPLTransImage):
            self.trans_image_tab.image = self.selected_image.trans_image

    @on_trait_change('selected_image')
    def update_viewer(self):

        if self.selected_row is not None:
            image_name = os.path.basename(self.selected_row.name)
            image_path = os.path.dirname(self.selected_row.name)
            data_dir = f"{image_path}/{image_name}-pyfibre-analysis/data/"
            filename = data_dir + image_name

            if isinstance(self.selected_image, SHGImage):

                self._update_shg_image_tabs(filename, image_name)

                if isinstance(self.selected_image, SHGPLImage):

                    self._update_pl_image_tabs(filename, image_name)
