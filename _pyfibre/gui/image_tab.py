import logging

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

log = logging.getLogger(__name__)


class ImageTab(BaseDisplayTab):
    """Standard image tab that just displays raw data for each
    labelled channel in a BaseMultiImage stack"""

    multi_image = Instance(BaseMultiImage)

    cmap = Function(reverse(binary))

    selected_label = Enum(values='image_labels')

    image_plot = Instance(Plot)

    image_dict = Dict(Str, Array)

    image_data = Instance(ArrayPlotData)

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

    def _image_data_default(self):
        return ArrayPlotData()

    def _image_plot_default(self):
        return Plot()

    def _get_image_labels(self):
        """Exposes list of keys for BaseMultiImage.image_dict."""
        if self.multi_image is None:
            return []
        return list(self.multi_image.image_dict.keys())

    @on_trait_change('brightness')
    def _brightness_updated(self):
        """Regenerate the selected image if runtime changes have been made
        """
        new_image = self._adjusted_image(
            self.multi_image.image_dict[self.selected_label])
        new_image = self.customise_image(new_image)
        self.image_data.set_data(self.selected_label, new_image)

    @on_trait_change('selected_label')
    def _selected_label_updated(self):
        """Regenerate plot if new image selected
        """
        self._refresh_image_plot()

    def _adjusted_image(self, image):
        gain = 1.0
        min_gamma = 0.001
        max_gamma = 2.0
        gamma = min_gamma + max_gamma * (1 - self.brightness)

        new_image = rescale_intensity(
            image,
            out_range=(0, 1)
        )
        new_image = adjust_gamma(
            new_image,
            gamma=gamma,
            gain=gain
        )
        return new_image

    def _refresh_image_dict(self):
        """Caches all images from the assigned BaseMultiImage trait for
        additional processing
        """
        if self.multi_image is None:
            image_dict = {}
        else:
            image_dict = {
                key: self._adjusted_image(image)
                for key, image in self.multi_image.image_dict.items()
            }
        self.image_dict = image_dict

    def _refresh_image_data(self):
        """Regenerates the ArrayPlotData object from all cached images
        with additional customisation options applied
        """
        self.image_data = ArrayPlotData(
            **{
                key: self.customise_image(image)
                for key, image in self.image_dict.items()
            }
        )

    def _refresh_image_plot(self):
        """Regenerates the chaco Plot object
        """
        if self.multi_image is None:
            self.image_plot = Plot()
            return

        plot = Plot(self.image_data)
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
        """Remove the currently assigned BaseMultiImage and reset the
        GUI components of the plot
        """
        self.multi_image = None
        self.update_tab()

    def update_tab(self):
        """Refreshes all cached image and plot objects
        """
        self._refresh_image_dict()
        self._refresh_image_data()
        self._refresh_image_plot()

    def customise_plot(self, plot):
        """Attach optional tools to plot"""
        pass

    def customise_image(self, image):
        """Overload to customize image"""
        return image


class TensorImageTab(ImageTab):

    def customise_image(self, image):
        """Convert each image into a tensor image"""
        new_image = create_tensor_image(image) * IMAGE_MAX
        return new_image.astype('uint8')


class NetworkImageTab(ImageTab):

    networks = List(nx.Graph)

    c_mode = Int(0)

    def customise_image(self, image):
        """Convert each image into a network image"""
        new_image = create_network_image(
            image,
            self.networks,
            c_mode=self.c_mode) * IMAGE_MAX
        return new_image.astype('uint8')

    def reset_tab(self):
        self.networks = []
        super().reset_tab()
