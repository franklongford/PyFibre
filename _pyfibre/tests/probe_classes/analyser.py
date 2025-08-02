import pandas as pd

from pyfibre.core.base_multi_image_analyser import BaseMultiImageAnalyser

from .multi_images import ProbeMultiImage


class ProbeAnalyser(BaseMultiImageAnalyser):

    database_names = ['probe']

    def __init__(self, *args, **kwargs):
        kwargs['multi_image'] = ProbeMultiImage()
        super().__init__(*args, **kwargs)

    def create_figures(self, *args, **kwargs):
        pass

    def create_metrics(self, *args, **kwargs):
        pass

    def image_analysis(self, *args, **kwargs):
        pass

    def save_databases(self, databases):
        pass

    def load_databases(self):
        return [pd.DataFrame() for _ in self.database_names]
