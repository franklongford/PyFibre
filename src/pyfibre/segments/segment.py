from dataclasses import dataclass

import numpy as np
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties

from pyfibre.tools.utilities import bbox_indices


@dataclass(kw_only=True)
class Segment:
    """Container for a scikit-image regionprops object
    representing a segmented area of an image"""

    region: RegionProperties

    @classmethod
    def from_array(
        cls, array: np.typing.NDArray, intensity_image: np.typing.NDArray | None = None
    ):
        """Deserialises numpy array to return an instance
        of the class"""
        labels = label(array.astype(np.uint32))
        region = regionprops(labels, intensity_image=intensity_image)[0]
        return cls(region=region)

    def to_array(self, shape: tuple[int, ...] | None = None):
        """Return the object state in a form that can be
        serialised as a numpy array"""
        indices = bbox_indices(self.region)
        if shape is None:
            shape = self.region.bbox[2:]
        array = np.zeros(shape, dtype=np.uint32)
        array[indices] += self.region.image

        return array
