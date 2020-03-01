import pandas as pd

from pyfibre.model.tools.metrics import (
    segment_shape_metrics, segment_texture_metrics)


class Cell:

    def __init__(self, segment=None, image=None):

        self.segment = segment
        self.image = image

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        if image is None:
            image = self.image

        if self.segment is None:
            raise AttributeError(
                'Cell.segment attribute must be assigned'
                'first'
            )

        database = pd.Series(dtype=object)

        shape_metrics = segment_shape_metrics(
            self.segment, tag='Cell')
        texture_metrics = segment_texture_metrics(
            self.segment, image=image, tag='Cell')

        database = database.append(shape_metrics, ignore_index=False)
        database = database.append(texture_metrics, ignore_index=False)

        return database
