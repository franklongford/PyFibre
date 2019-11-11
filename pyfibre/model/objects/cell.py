import pandas as pd

from pyfibre.model.tools.analysis import segment_analysis


class Cell:

    def __init__(self, segment=None, image=None):

        self.segment = segment
        self.image = image

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        database = pd.Series()

        if image is not None:
            segment_metrics = segment_analysis(
                self.segment, image=image, tag='Cell')

        else:
            segment_metrics = segment_analysis(
                self.segment, tag='Cell')

        database = database.append(segment_metrics, ignore_index=False)

        return database

