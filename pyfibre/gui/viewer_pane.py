from pyface.tasks.api import TraitsTaskPane
from pyface.api import ImageResource
from pyface.ui_traits import convert_image
from chaco.api import ArrayPlotData, Plot
from enable.api import ComponentEditor

from traits.api import (
    HasTraits, Instance, Unicode, List, on_trait_change,
    ArrayOrNone, Property, Array
)
from traitsui.api import (
    View, VGroup, Group, UItem, ImageEditor, HGroup,
    Spring, Image, Item, ListEditor
)

from pyfibre.io.multi_image import MultiLayerImage


class ImageTab(HasTraits):

    image = ArrayOrNone

    label = Unicode()

    plot = Property(Instance(Plot),
                    depends_on='image')

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

    segment_tab = Instance(ImageTab)

    fibre_tab = Instance(ImageTab)

    cell_tab = Instance(ImageTab)

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
                self.pl_image_tab]

    def _shg_image_tab_default(self):
        if self.selected_image is not None:
            return ImageTab(
                label='SHG',
                image=self.selected_image.image_shg)
        else:
            return ImageTab(label='SHG')

    def _pl_image_tab_default(self):
        if self.selected_image is not None:
            return ImageTab(
                label='PL',
                image=self.selected_image.image_pl)
        else:
            return ImageTab(label='PL')

    @on_trait_change('selected_image')
    def update_viewer(self):
        print('update_viewer called')
        self.shg_image_tab.image = self.selected_image.image_shg
        self.pl_image_tab.image = self.selected_image.image_pl