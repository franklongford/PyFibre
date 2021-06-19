from chaco.array_plot_data import ArrayPlotData
from traits.api import List

from pyfibre.gui.metric_tab import ImageMetricTab
from pyfibre.model.core.base_segment import BaseSegment
from pyfibre.model.tools.figures import create_region_image


class SegmentImageTab(ImageMetricTab):

    segments = List(BaseSegment)

    def _region_image(self, image):
        return create_region_image(
            image,
            [segment.region for segment in self.segments])

    def _update_image_data(self):
        """Convert each image into a segment image"""
        if self.multi_image is None:
            image_dict = {}
        else:
            image_dict = {
                label: self._region_image(image).astype('uint8')
                for label, image in self.multi_image.image_dict.items()}
        self.image_data = ArrayPlotData(**image_dict)

    def reset_tab(self):
        self.segments = []
        super().reset_tab()
