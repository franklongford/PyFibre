import logging
import os
from chaco.api import ArrayPlotData, Plot
from chaco.default_colormaps import binary, reverse
from enable.api import ComponentEditor

from pyface.tasks.api import TraitsTaskPane

from traits.api import (
    HasTraits, Instance, Unicode, List, on_trait_change,
    ArrayOrNone, Property, Array, Function
)
from traitsui.api import (
    View, VGroup, Group, UItem, ImageEditor, HGroup,
    Spring, Image, Item, ListEditor
)

from pyfibre.io.multi_image import MultiLayerImage
from pyfibre.io.segment_io import load_segment
from pyfibre.io.network_io import load_network
from pyfibre.io.database_io import check_file_name
from pyfibre.model.tools.figures import (
    create_tensor_image, create_region_image, create_network_image
)
from pyfibre.model.tools.extraction import network_extraction
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

    def _get_plot(self):
        if self.image is not None:
            plot_data = ArrayPlotData(
                image_data=self.image)

            plot = Plot(plot_data)

            if self.image.ndim == 2:
                plot.img_plot("image_data",
                              colormap=self.cmap)
            elif self.image.ndim == 3:
                plot.img_plot("image_data")

            return plot


class MetricTab(HasTraits):
    pass


class ViewerPane(TraitsTaskPane):

    id = 'pyfibre.viewer_pane'

    name = 'Viewer Pane'

    image_tab_list = List(Instance(ImageTab))

    shg_image_tab = Instance(ImageTab)

    pl_image_tab = Instance(ImageTab)

    tran_image_tab = Instance(ImageTab)

    tensor_tab = Instance(ImageTab)

    network_tab = Instance(ImageTab)

    fibre_tab = Instance(ImageTab)

    fibre_segment_tab = Instance(ImageTab)

    cell_segment_tab = Instance(ImageTab)

    metric_tab = Instance(MetricTab)

    # Properties
    selected_image = Instance(MultiLayerImage)

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

    def _image_tab_list_default(self):

        return [self.shg_image_tab,
                self.pl_image_tab,
                self.tran_image_tab,
                self.tensor_tab,
                self.network_tab,
                self.fibre_tab,
                self.fibre_segment_tab,
                self.cell_segment_tab]

    def _shg_image_tab_default(self):
        image_tab = ImageTab(label='SHG Image')
        if self.selected_image is not None:
            image_tab.image = self.selected_image.image_shg
        return image_tab

    def _pl_image_tab_default(self):
        image_tab = ImageTab(label='PL Image')
        if self.selected_image is not None:
            image_tab.image = self.selected_image.image_pl
        return image_tab

    def _tran_image_tab_default(self):
        image_tab = ImageTab(label='Transmission Image')
        if self.selected_image is not None:
            image_tab.image = self.selected_image.image_tran
        return image_tab

    def _tensor_tab_default(self):
        image_tab = ImageTab(label='Tensor Image')
        if self.selected_image is not None:
            tensor_image = create_tensor_image(
                self.selected_image.image_shg
            ) * 255.999
            image_tab.image = tensor_image
        return image_tab

    def _network_tab_default(self):
        image_tab = ImageTab(label='Network')
        if self.selected_image is not None:
            image_name = os.path.basename(self.selected_image.file_path)
            image_path = os.path.dirname(self.selected_image.file_path)
            data_dir = image_path + '/data/'
            try:
                networks = load_network(
                    data_dir + image_name, "network")
                image_network_overlay = create_network_image(
                    self.selected_image.image_shg, networks, 0)
                self.network_tab.image = image_network_overlay.astype('uint8')
            except (IOError, EOFError):
                logger.info(
                    "Unable to display network for {}".format(image_name))
        return image_tab

    def _fibre_tab_default(self):
        image_tab = ImageTab(label='Fibres')
        return image_tab

    def _fibre_segment_tab_default(self):
        image_tab = ImageTab(label='Fibre Segments')
        return image_tab

    def _cell_segment_tab_default(self):
        image_tab = ImageTab(label='Cell Segments')
        return image_tab

    @on_trait_change('selected_image')
    def update_viewer(self):

        image_name = os.path.basename(self.selected_image.file_path)
        image_path = os.path.dirname(self.selected_image.file_path)
        data_dir = image_path + '/data/'

        if self.selected_image.shg_analysis:
            self.shg_image_tab.image = self.selected_image.image_shg
            tensor_image = create_tensor_image(
                self.selected_image.image_shg
            ) * 255.999
            self.tensor_tab.image = tensor_image.astype('uint8')

            try:
                network = load_network(data_dir + image_name, "network")
            except (IOError, EOFError):
                logger.info("Unable to display network for {}".format(image_name))
            else:
                networks, networks_red, fibres = network_extraction(network)
                network_image = create_network_image(
                    self.selected_image.image_shg,
                    networks,
                    0) * 255.999
                self.network_tab.image = network_image.astype('uint8')

                fibres = flatten_list(fibres)

                fibre_image = create_network_image(
                    self.selected_image.image_shg,
                    fibres,
                    1) * 255.999
                self.fibre_tab.image = fibre_image.astype('uint8')

            try:
                segments = load_segment(data_dir + image_name, "fibre_segment")
            except (AttributeError, IOError, EOFError):
                self.update_log("Unable to display fibre segments for {}".format(image_name))
            else:
                segment_image = create_region_image(
                    self.selected_image.image_shg,
                    segments) * 255.999
                self.fibre_segment_tab.image = segment_image.astype('uint8')

        if self.selected_image.pl_analysis:

            self.pl_image_tab.image = self.selected_image.image_pl
            self.tran_image_tab.image = self.selected_image.image_tran

            try:
                segments = load_segment(data_dir + image_name, "cell_segment")
            except (AttributeError, IOError, EOFError):
                self.update_log("Unable to display cell segments for {}".format(image_name))
            else:
                segment_image = create_region_image(
                    self.selected_image.image_pl,
                    segments) * 255.999
                self.cell_segment_tab.image = segment_image.astype('uint8')
