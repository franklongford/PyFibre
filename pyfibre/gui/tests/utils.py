from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin

from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.gui.pyfibre_plugin import PyFibrePlugin
from pyfibre.gui.pyfibre_main_task import PyFibreMainTask


class DummyPyFibreGUI(PyFibreGUI):

    def __init__(self):

        plugins = [CorePlugin(), TasksPlugin()]
        super(PyFibreGUI, self).__init__(plugins=plugins)

    def run(self):
        """Run the application (dummy class:
        does nothing in this case)."""
        pass


class ProbePyFibrePlugin(PyFibrePlugin):

    def _create_main_task(self):
        pyfibre_task = PyFibreMainTask()
        return pyfibre_task


class ProbePyFibreGUI(PyFibreGUI):

    def __init__(self):

        plugins = [CorePlugin(), TasksPlugin(),
                   ProbePyFibrePlugin()]

        super(ProbePyFibreGUI, self).__init__(plugins=plugins)

        # 'Run' the application by creating windows without an event loop
        self.run = self._create_windows