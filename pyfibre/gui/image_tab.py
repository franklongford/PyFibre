import networkx as nx

from chaco.api import ArrayPlotData, Plot
from chaco.tools.zoom_tool import ZoomTool
from chaco.tools.pan_tool import PanTool
from chaco.default_colormaps import binary, reverse
from enable.component_editor import ComponentEditor
from traits.api import (
    Instance, Function, List, Int, Property, Str, Dict, Enum,
    cached_property)
from traitsui.api import Item, View, EnumEditor

from pyfibre.core.base_multi_image import BaseMultiImage
from pyfibre.core.base_multi_image_viewer import BaseDisplayTab
from pyfibre.model.tools.figures import (
    create_tensor_image, create_network_image)


class ImageTab(BaseDisplayTab):
    """Standard image tab that just displays raw data for each
    labelled channel in a BaseMultiImage stack"""

    multi_image = Instance(BaseMultiImage)

    cmap = Function(reverse(binary))

    selected_label = Enum(values='image_labels')

    image_plot = Property(
        Instance(Plot),
        depends_on='image_data,'
                   'selected_label')

    image_data = Property(
        Instance(ArrayPlotData),
        depends_on='_image_dict'
    )

    image_labels = Property(
        List(Str),
        depends_on='_image_dict'
    )

    _image_dict = Property(
        Dict, depends_on='multi_image.image_dict'
    )

    trait_view = View(
            Item('image_plot',
                 editor=ComponentEditor(),
                 show_label=False),
            Item('selected_label',
                 editor=EnumEditor(
                     name='object.image_labels'),
                 style='simple'),
            resizable=True
        )

    @cached_property
    def _get_image_plot(self):
        plot = Plot(self.image_data)

        if self.multi_image is None:
            return plot

        kwargs = {"origin": 'top left',
                  'axis': 'off'}

        if self.multi_image.ndim == 2:
            kwargs['colormap'] = self.cmap

        plot.img_plot(
            self.selected_label,
            **kwargs)

        # Attach some tools to the plot
        plot.tools.append(
            PanTool(plot, constrain_key="shift"))
        plot.overlays.append(ZoomTool(
            component=plot,
            tool_mode="box",
            always_on=False))

        return plot

    def _get__image_dict(self):
        """Simply exposes the BaseMultiImage.image_dict attribute."""
        if self.multi_image is not None:
            return self.multi_image.image_dict
        return {}

    def _get_image_labels(self):
        """Exposes list of keys for BaseMultiImage.image_dict."""
        return list(self._image_dict.keys())

    def _get_image_data(self):
        return ArrayPlotData(**self._image_dict)

    def customise_plot(self, plot):
        """Attach optional tools to plot"""
        pass


class TensorImageTab(ImageTab):

    def _tensor_image(self, image):
        return create_tensor_image(image)

    def _get_image_data(self):
        """Convert each image into a tensor image"""
        image_dict = {
            label: self._tensor_image(image).astype('uint8')
            for label, image in self._image_dict.items()}
        return ArrayPlotData(**image_dict)


class NetworkImageTab(ImageTab):

    networks = List(nx.Graph)

    c_mode = Int(0)

    image_data = Property(
        Instance(ArrayPlotData),
        depends_on='_image_dict,networks'
    )

    def _network_image(self, image):
        return create_network_image(
            image,
            self.networks,
            c_mode=self.c_mode)

    def _get_image_data(self):
        """Convert each image into a network image"""
        image_dict = {
            label: self._network_image(image).astype('uint8')
            for label, image in self._image_dict.items()}
        return ArrayPlotData(**image_dict)
