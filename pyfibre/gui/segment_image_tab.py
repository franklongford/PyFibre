from chaco.array_plot_data import ArrayPlotData
from skimage.measure._regionprops import RegionProperties
from traits.trait_types import List, Any
from traits.traits import Property

from pyfibre.gui.image_tab import ImageTab
from pyfibre.model.objects.base_segment import BaseSegment
from pyfibre.model.tools.figures import create_region_image


class SegmentImageTab(ImageTab):

    segments = List(BaseSegment)

    regions = Property(
        List(RegionProperties), depends_on='segments')

    plot_data = Property(
        Any, depends_on='_image_dict,regions'
    )

    def _get_regions(self):
        return [segment.region for segment in self.segments]

    def _region_image(self, image):
        return create_region_image(
            image,
            self.regions) * 255.999

    def _get_plot_data(self):
        """Convert each image into a segment image"""
        image_dict = {
            label: self._region_image(image).astype('uint8')
            for label, image in self._image_dict.items()}
        return ArrayPlotData(**image_dict)
