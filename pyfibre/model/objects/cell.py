import pandas as pd

from pyfibre.model.tools.metrics import segment_metrics


class Cell:

    def __init__(self, segment=None, image=None):

        self.segment = segment
        self.image = image

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        database = pd.Series()

        if image is not None:
            metrics = segment_metrics(
                self.segment, image=image, tag='Cell')

        else:
            metrics = segment_metrics(
                self.segment, tag='Cell')

        database = database.append(metrics, ignore_index=False)

        return database

