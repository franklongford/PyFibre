from chaco.array_plot_data import ArrayPlotData
from skimage.measure._regionprops import RegionProperties
from traits.api import List, Any, Property

from pyfibre.gui.metric_tab import ImageMetricTab
from pyfibre.model.core.base_segment import BaseSegment
from pyfibre.model.tools.figures import create_region_image


class SegmentImageTab(ImageMetricTab):

    segments = List(BaseSegment)

    regions = Property(
        List(RegionProperties), depends_on='segments')

    image_data = Property(
        Any, depends_on='_image_dict,regions'
    )

    def _get_regions(self):
        return [segment.region for segment in self.segments]

    def _region_image(self, image):
        return create_region_image(
            image,
            self.regions)

    def _get_image_data(self):
        """Convert each image into a segment image"""
        image_dict = {
            label: self._region_image(image).astype('uint8')
            for label, image in self._image_dict.items()}
        return ArrayPlotData(**image_dict)
