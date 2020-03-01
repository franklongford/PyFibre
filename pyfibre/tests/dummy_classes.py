import numpy as np
from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin

from pyfibre.cli.pyfibre_cli import PyFibreCLI
from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.model.tools.fibre_assigner import Fibre


class DummyFibre(Fibre):

    def __init__(self, fibre_l=None, euclid_l=None, direction=None,
                 *args, **kwargs):
        super(DummyFibre, self).__init__(
            nodes=[0, 1], *args, **kwargs
        )

        if fibre_l is None or euclid_l is None:
            euclid_l = np.random.random_sample()
            fibre_l = euclid_l + np.random.random_sample()

        if direction is None:
            direction = [np.random.random_sample(),
                         np.random.random_sample()]

        super(DummyFibre, self).__init__(
            nodes=[], *args, **kwargs
        )


class DummyPyFibreCLI(PyFibreCLI):

    def run(self, file_name):
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