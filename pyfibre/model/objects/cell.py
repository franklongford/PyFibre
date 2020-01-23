import pandas as pd

from pyfibre.model.tools.analysis import (
    segment_shape_analysis, segment_texture_analysis)


class Cell:

    def __init__(self, segment=None, image=None):

        self.segment = segment
        self.image = image

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        database = pd.Series()

        shape_metrics = segment_shape_analysis(
            self.segment, tag='Cell')
        texture_metrics = segment_texture_analysis(
            self.segment, image=image, tag='Cell')

        database = database.append(shape_metrics, ignore_index=False)
        database = database.append(texture_metrics, ignore_index=False)

        return database

