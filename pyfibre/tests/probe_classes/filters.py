import numpy as np

from pyfibre.model.tools.base_bd_filter import BaseBDFilter


class ProbeBDFilter(BaseBDFilter):

    def cellular_classifier(self, label_image, centres, **kwargs):
        return np.ones(len(centres)), 1.0
