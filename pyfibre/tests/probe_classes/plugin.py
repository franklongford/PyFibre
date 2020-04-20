from pyfibre.gui.pyfibre_main_task import PyFibreMainTask
from pyfibre.gui.pyfibre_plugin import PyFibrePlugin


class ProbePyFibrePlugin(PyFibrePlugin):

    def _create_main_task(self):
        pyfibre_task = PyFibreMainTask()
        return pyfibre_task
