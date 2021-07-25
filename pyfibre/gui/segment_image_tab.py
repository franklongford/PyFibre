from chaco.array_plot_data import ArrayPlotData
from traits.api import List

from pyfibre.gui.metric_tab import ImageMetricTab
from pyfibre.model.core.base_segment import BaseSegment
from pyfibre.model.tools.figures import create_region_image
from pyfibre.utilities import IMAGE_MAX


class SegmentImageTab(ImageMetricTab):

    segments = List(BaseSegment)

    def customise_image(self, image):
        new_image = create_region_image(
            image,
            [segment.region for segment in self.segments]) * IMAGE_MAX
        return new_image.astype('uint8')

    def reset_tab(self):
        self.segments = []
        super().reset_tab()
