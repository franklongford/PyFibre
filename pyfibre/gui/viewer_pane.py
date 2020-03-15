import logging
import os

import networkx as nx
from skimage.measure._regionprops import RegionProperties

from chaco.api import ArrayPlotData, Plot
from chaco.default_colormaps import binary, reverse
from chaco.tools.api import PanTool, ZoomTool
from enable.api import ComponentEditor
from pyface.tasks.api import TraitsTaskPane
from traits.api import (
    HasTraits, Instance, Unicode, List, on_trait_change,
    ArrayOrNone, Property, Function, Int
)
from traitsui.api import (
    View, Group, Item, ListEditor
)

from pyfibre.io.object_io import (
    load_cells, load_fibre_networks, load_fibres)
from pyfibre.io.shg_pl_reader import SHGPLTransReader
from pyfibre.gui.file_display_pane import TableRow
from pyfibre.model.objects.multi_image import MultiImage
from pyfibre.model.tools.figures import (
    create_tensor_image, create_region_image, create_network_image
)
from pyfibre.utilities import flatten_list

logger = logging.getLogger(__name__)


class ImageTab(HasTraits):

    image = ArrayOrNone()

    label = Unicode()

    plot = Property(Instance(Plot),
                    depends_on='image')

    cmap = Function(reverse(binary))

    traits_view = View(
        Item('plot',
             editor=ComponentEditor(),
             show_label=False),
        resizable=True
    )

    def _plot_image(self, plot):

        if self.image.ndim == 2:
            img_plot = plot.img_plot(
                "image_data",
                origin='top left',
                colormap=self.cmap,
                axis='off')[0]
        elif self.image.ndim == 3:
            img_plot = plot.img_plot(
                "image_data",
                origin='top left',
                axis='off')[0]

            # Attach some tools to the plot
            plot.tools.append(PanTool(
                plot, constrain_key="shift"))
            plot.overlays.append(ZoomTool(
                component=plot,
                tool_mode="box",
                always_on=False))

    def _get_plot(self):
        if self.image is not None:
            plot_data = ArrayPlotData(
                image_data=self.image)

            plot = Plot(plot_data)

            self._plot_image(plot)

            return plot


class NetworkImageTab(ImageTab):

    networks = List(nx.Graph)

    c_mode = Int(0)

    def _get_plot(self):
        if self.image is not None:

            image_network_overlay = create_network_image(
                self.image,
                self.networks,
                c_mode=self.c_mode) * 255.999

            plot_data = ArrayPlotData(
                image_data=image_network_overlay.astype('uint8'))

            plot = Plot(plot_data)

            self._plot_image(plot)

            return plot


class SegmentImageTab(ImageTab):

    segments = List(RegionProperties)

    def _get_plot(self):
        if self.image is not None:
            segment_image = create_region_image(
                self.image,
                self.segments) * 255.999

            plot_data = ArrayPlotData(
                image_data=segment_image.astype('uint8'))

            plot = Plot(plot_data)

            self._plot_image(plot)

            return plot


class MetricTab(HasTraits):
    pass


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

    @on_trait_change('selected_image')
    def update_viewer(self):

        if self.selected_row is not None:
            image_name = os.path.basename(self.selected_row.name)
            image_path = os.path.dirname(self.selected_row.name)
            data_dir = f"{image_path}/{image_name}-pyfibre-analysis/data/"
            filename = data_dir + image_name

            self.shg_image_tab.image = self.selected_image.shg_image
            tensor_image = create_tensor_image(
                self.selected_image.shg_image
            ) * 255.999
            self.tensor_tab.image = tensor_image.astype('uint8')

            try:
                fibre_networks = load_fibre_networks(filename)
            except (IOError, EOFError):
                logger.info("Unable to display network for {}".format(image_name))
            else:
                fibre_segments = [fibre_network.region for fibre_network in fibre_networks]
                networks = [fibre_network.graph for fibre_network in fibre_networks]

                self.network_tab.networks = networks
                self.network_tab.image = self.selected_image.shg_image

                self.fibre_segment_tab.segments = fibre_segments
                self.fibre_segment_tab.image = self.selected_image.shg_image

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

            self.pl_image_tab.image = self.selected_image.pl_image
            self.trans_image_tab.image = self.selected_image.trans_image

            try:
                cells = load_cells(
                    filename, image=self.selected_image.pl_image)
            except (AttributeError, IOError, EOFError):
                logger.debug("Unable to display cell segments for {}".format(image_name))
            else:
                cell_segments = [cell.region for cell in cells]

                self.cell_segment_tab.segments = cell_segments
                self.cell_segment_tab.image = self.selected_image.pl_image
