from pyfibre.model.core.base_analyser import BaseAnalyser

from .multi_images import ProbeMultiImage


class ProbeAnalyser(BaseAnalyser):

    def __init__(self, *args, **kwargs):
        kwargs['multi_image'] = ProbeMultiImage()
        super().__init__(*args, **kwargs)

    def create_figures(self, *args, **kwargs):
        pass

    def create_metrics(self, *args, **kwargs):
        pass

    def image_analysis(self, *args, **kwargs):
        pass
