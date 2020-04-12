import pandas as pd
import numpy as np
from skimage.measure import label, regionprops

from pyfibre.model.tools.metrics import (
    region_shape_metrics, region_texture_metrics)

from .abc_pyfibre_object import ABCPyFibreObject


class BaseSegment(ABCPyFibreObject):
    """Container for a scikit-image regionprops object
    representing a segmented area of an image"""

    _tag = None

    def __init__(self, region=None):
        self.region = region

    @property
    def _shape_tag(self):
        if self._tag is None:
            return 'Segment'
        return f'{self._tag} Segment'

    @classmethod
    def from_json(cls, data):
        """Deserialises JSON data dictionary to return an instance
        of the class"""
        raise NotImplementedError(
            f'from_json method not supported for {cls.__class__}')

    def to_json(self):
        """Serialises instance into a dictionary able to be dumped as a
        JSON file"""
        raise NotImplementedError(
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
        minr, minc, maxr, maxc = self.region.bbox
        if shape is None:
            shape = (maxr, maxc)
        indices = np.mgrid[minr:maxr, minc:maxc]
        array = np.zeros(shape, dtype=np.int)
        array[(indices[0], indices[1])] += self.region.image

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
