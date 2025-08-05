import numpy as np

from pyfibre.tools.base_kmeans_filter import BaseKmeansFilter


class ProbeKmeansFilter(BaseKmeansFilter):

    def cellular_classifier(self, label_image, centres, **kwargs):
        return np.ones(len(centres)), 1.0
