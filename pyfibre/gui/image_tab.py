import networkx as nx
from skimage.measure._regionprops import RegionProperties

from chaco.api import (
    ArrayPlotData, reverse, binary, Plot
)
from chaco.tools.zoom_tool import ZoomTool
from chaco.tools.pan_tool import PanTool
from enable.component_editor import ComponentEditor
from traits.api import (
    HasTraits, ArrayOrNone, Unicode, Instance, Function,
    List, Int, Property)
from traitsui.api import Item, View

from pyfibre.model.objects.base_segment import BaseSegment
from pyfibre.model.tools.figures import (
    create_network_image, create_region_image)


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
            plot.img_plot(
                "image_data",
                origin='top left',
                colormap=self.cmap,
                axis='off')
        elif self.image.ndim == 3:
            plot.img_plot(
                "image_data",
                origin='top left',
                axis='off')

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

    segments = List(BaseSegment)

    regions = Property(
        List(RegionProperties), depends_on='segments')

    def _get_regions(self):
        return [segment.region for segment in self.segments]

    def _get_plot(self):
        if self.image is not None:
            segment_image = create_region_image(
                self.image,
                self.regions) * 255.999

            plot_data = ArrayPlotData(
                image_data=segment_image.astype('uint8'))

            plot = Plot(plot_data)

            self._plot_image(plot)

            return plot


class MetricTab(HasTraits):
    pass
