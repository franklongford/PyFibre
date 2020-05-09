from abc import abstractmethod
import os

from traits.api import ABCHasTraits, Instance

from pyfibre.model.multi_image.base_multi_image import BaseMultiImage


class BaseAnalyser(ABCHasTraits):
    """Class that provides analysis instructions for a particular
    BaseMultiImage class"""

    multi_image = Instance(BaseMultiImage)

    @abstractmethod
    def image_analysis(self, *args, **kwargs):
        """Perform analysis on data"""

    @abstractmethod
    def create_metrics(self, *args, **kwargs):
        """Create metrics from multi-image components that can be
        generated upon end of analysis"""

    @abstractmethod
    def create_figures(self, *args, **kwargs):
        """Create figures from multi-image components that can be
        generated upon end of analysis"""

    @property
    def analysis_path(self):
        return os.path.join(
            self.multi_image.path,
            '-'.join([self.multi_image.name, 'pyfibre', 'analysis'])
        )

    def make_directories(self):
        """Creates required directories for analysis"""
        if not os.path.exists(self.analysis_path):
            os.mkdir(self.analysis_path)
