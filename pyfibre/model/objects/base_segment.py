import copy

import pandas as pd

from pyfibre.model.tools.metrics import (
    region_shape_metrics, region_texture_metrics)
from pyfibre.io.utilities import pop_under_recursive


class BaseSegment:
    """Container for a scikit-image regionprops object
    representing a segmented area of an image"""

    _tag = None

    def __init__(self, region=None, image=None):

        self.region = region

        if image is None:
            if self.region.intensity_image is not None:
                image = region.intensity_image

        self.image = image

    def __getstate__(self):
        """Return the object state in a form that can be
        serialised as a JSON file"""
        state = pop_under_recursive(copy.copy(self.__dict__))
        state.pop('image', None)

        return state

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        if self.region is None:
            raise AttributeError(
                'BaseSegment.segment attribute must be assigned'
                'first'
            )

        if image is None:
            image = self.image
        if self.region._intensity_image is not None:
            image = self.region.intensity_image

        database = pd.Series(dtype=object)

        shape_metrics = region_shape_metrics(
            self.region, tag=self._tag)
        texture_metrics = region_texture_metrics(
            self.region, image=image, tag=self._tag)

        database = database.append(shape_metrics, ignore_index=False)
        database = database.append(texture_metrics, ignore_index=False)

        return database