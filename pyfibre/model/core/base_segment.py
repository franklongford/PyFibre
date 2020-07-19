from abc import abstractmethod

import pandas as pd
import numpy as np
from skimage.measure import label, regionprops

from pyfibre.model.tools.metrics import (
    region_shape_metrics, region_texture_metrics)
from pyfibre.utilities import NotSupportedError
from pyfibre.model.tools.utilities import bbox_indices

from .base_pyfibre_object import BasePyFibreObject


class BaseSegment(BasePyFibreObject):
    """Container for a scikit-image regionprops object
    representing a segmented area of an image"""

    def __init__(self, region=None):
        self.region = region

    @property
    def _shape_tag(self):
        return ' '.join([self.tag, 'Segment'])

    @property
    def tag(self):
        return self.get_tag()

    @abstractmethod
    def get_tag(self):
        """String representing class type"""

    @classmethod
    def from_json(cls, data):
        """Deserialises JSON data dictionary to return an instance
        of the class"""
        raise NotSupportedError(
            f'from_json method not supported for {cls.__class__}')

    def to_json(self):
        """Serialises instance into a dictionary able to be dumped as a
        JSON file"""
        raise NotSupportedError(
            f'to_json method not supported for {self.__class__}')

    @classmethod
    def from_array(cls, array, intensity_image=None):
        """Deserialises numpy array to return an instance
        of the class"""
        labels = label(array.astype(np.int))
        region = regionprops(
            labels, intensity_image=intensity_image)[0]
        return cls(region=region)

    def to_array(self, shape=None):
        """Return the object state in a form that can be
        serialised as a numpy array"""
        indices = bbox_indices(self.region)
        if shape is None:
            shape = self.region.bbox[2:]
        array = np.zeros(shape, dtype=np.int)
        array[indices] += self.region.image

        return array

    def generate_database(self, image_tag=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        if self.region is None:
            raise AttributeError(
                'BaseSegment.region attribute must be assigned'
                'first'
            )

        if image_tag is None:
            texture_tag = self._shape_tag
        else:
            texture_tag = ' '.join([self._shape_tag, image_tag])

        database = pd.Series(dtype=object)

        shape_metrics = region_shape_metrics(
            self.region, tag=self._shape_tag)

        texture_metrics = region_texture_metrics(
            self.region, tag=texture_tag)

        database = database.append(shape_metrics, ignore_index=False)
        database = database.append(texture_metrics, ignore_index=False)

        return database
