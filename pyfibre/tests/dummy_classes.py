from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin

from pyfibre.cli.app.pyfibre_app import PyFibreApplication
from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.model.core.base_graph import BaseGraph
from pyfibre.model.core.base_graph_segment import BaseGraphSegment
from pyfibre.model.core.base_multi_image import BaseMultiImage
from pyfibre.model.core.base_multi_image_analyser import BaseMultiImageAnalyser


class DummyMultiImage(BaseMultiImage):

    def preprocess_images(self):
        pass

    @classmethod
    def verify_stack(cls, image_stack):
        return True

    def segmentation_algorithm(self, *args, **kwargs):
        pass


class DummyAnalyser(BaseMultiImageAnalyser):

    def create_figures(self, *args, **kwargs):
        pass

    def create_metrics(self, *args, **kwargs):
        pass

    def image_analysis(self, *args, **kwargs):
        pass


class DummyGraph(BaseGraph):

    def generate_database(self, *args, **kwargs):
        pass


class DummyGraphSegment(BaseGraphSegment):

    def generate_database(self, *args, **kwargs):
        pass


class DummyPyFibreApplication(PyFibreApplication):

    def _run_pyfibre(self):
        """Run the application (dummy class:
        does nothing in this case)."""
        pass


class DummyPyFibreGUI(PyFibreGUI):

    def __init__(self):

        plugins = [CorePlugin(), TasksPlugin()]
        super(PyFibreGUI, self).__init__(plugins=plugins)

    def run(self):
        """Run the application (dummy class:
        does nothing in this case)."""
        pass
