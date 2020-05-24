from pyfibre.gui.pyfibre_main_task import PyFibreMainTask
from pyfibre.gui.pyfibre_plugin import PyFibreGUIPlugin
from pyfibre.core.base_pyfibre_plugin import BasePyFibrePlugin

from .factories import ProbeMultiImageFactory


class ProbePyFibrePlugin(BasePyFibrePlugin):

    def get_multi_image_factories(self):
        return [ProbeMultiImageFactory]


class ProbePyFibreGUIPlugin(PyFibreGUIPlugin):

    def _create_main_task(self):
        return PyFibreMainTask()
