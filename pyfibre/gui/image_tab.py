import networkx as nx
from skimage.exposure import adjust_gamma, rescale_intensity

from chaco.api import ArrayPlotData, Plot
from chaco.tools.zoom_tool import ZoomTool
from chaco.tools.pan_tool import PanTool
from chaco.default_colormaps import binary, reverse
from enable.component_editor import ComponentEditor
from traits.api import (
    Instance, Function, List, Int, Property, Str, Enum, Range,
    Array, Dict, on_trait_change)
from traitsui.api import Item, View, HGroup, VGroup

from pyfibre.core.base_multi_image import BaseMultiImage
from pyfibre.core.base_multi_image_viewer import BaseDisplayTab
from pyfibre.model.tools.figures import (
    create_tensor_image, create_network_image)
from pyfibre.utilities import IMAGE_MAX


class ImageTab(BaseDisplayTab):
    """Standard image tab that just displays raw data for each
    labelled channel in a BaseMultiImage stack"""

    multi_image = Instance(BaseMultiImage)

    cmap = Function(reverse(binary))

    selected_label = Enum(values='image_labels')

    image_plot = Instance(Plot)

    image_data = Dict(Str, Array)

    image_labels = Property(
        List(Str),
        depends_on='multi_image.image_dict'
    )

    brightness = Range(low=0.0, high=1.0, value=0.5)

    trait_view = View(
        VGroup(
            Item('image_plot',
                 editor=ComponentEditor(),
                 show_label=False),
            HGroup(
                Item('selected_label',
                     style='simple'),
                Item('brightness')
            )
        ),
        resizable=True
    )

    def _image_plot_default(self):
        return Plot()

    def _get_image_labels(self):
        """Exposes list of keys for BaseMultiImage.image_dict."""
        if self.multi_image is None:
            return []
        return list(self.multi_image.image_dict.keys())

    @on_trait_change('selected_label')
    def _selected_label_updated(self):
        self._update_image_plot()
        self._brightness_updated()

    @on_trait_change('brightness')
    def _brightness_updated(self):
        if self.multi_image is None:
            return
        gain = 1.0
        min_gamma = 0.001
        max_gamma = 2.0
        gamma = min_gamma + max_gamma * (1 - self.brightness)

        new_image = rescale_intensity(
            self.image_data[self.selected_label],
            out_range=(0, 1)
        )
        new_image = adjust_gamma(
            new_image,
            gamma=gamma,
            gain=gain
        )
        self.image_plot.data.update_data(
            {
                self.selected_label: new_image
            }
        )

    def _update_image_data(self):
        if self.multi_image is None:
            image_data = {}
        else:
            image_data = self.multi_image.image_dict
        self.image_data = image_data

    def _update_image_plot(self):
        if self.multi_image is None:
            self.image_plot = Plot()
            return

        plot = Plot(ArrayPlotData(**self.image_data.copy()))
        kwargs = {"origin": 'top left',
                  'axis': 'off'}
        if self.multi_image.ndim == 2:
            kwargs['colormap'] = self.cmap

        if self.selected_label:
            plot.img_plot(
                self.selected_label,
                name=self.selected_label,
                **kwargs)

        # Attach some tools to the plot
        plot.tools.append(
            PanTool(plot, constrain_key="shift"))
        plot.overlays.append(ZoomTool(
            component=plot,
            tool_mode="box",
            always_on=False))

        self.image_plot = plot

    def reset_tab(self):
        self.multi_image = None
        self.update_tab()

    def update_tab(self):
        """Provide additional customisation to chaco Plot object
        generated by this class."""
        self._update_image_data()
        self._update_image_plot()
        self._brightness_updated()

    def customise_plot(self, plot):
        """Attach optional tools to plot"""
        pass


class TensorImageTab(ImageTab):

    def _tensor_image(self, image):
        return create_tensor_image(image) * IMAGE_MAX

    def _update_image_data(self):
        """Convert each image into a tensor image"""
        if self.multi_image is None:
            image_data = {}
        else:
            image_data = {
                label: self._tensor_image(image).astype('uint8')
                for label, image in self.multi_image.image_dict.items()}
        self.image_data = image_data


class NetworkImageTab(ImageTab):

    networks = List(nx.Graph)

    c_mode = Int(0)

    def _network_image(self, image):
        return create_network_image(
            image,
            self.networks,
            c_mode=self.c_mode) * IMAGE_MAX

    def _update_image_data(self):
        """Convert each image into a network image"""
        if self.multi_image is None:
            image_data = {}
        else:
            image_data = {
                label: self._network_image(image).astype('uint8')
                for label, image in self.multi_image.image_dict.items()}
        self.image_data = image_data

    def reset_tab(self):
        self.networks = []
        super().reset_tab()
