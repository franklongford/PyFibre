from envisage.api import Plugin
from envisage.ui.tasks.api import TaskFactory
from traits.api import List

from pyfibre.gui.pyfibre_main_task import PyFibreMainTask


class PyFibrePlugin(Plugin):
    """ The Plugin containing the PyFibre UI. This contains the
    factories which create the Tasks (currently Main)"""

    TASKS = 'envisage.ui.tasks.tasks'

    id = 'pyfibre.pyfibre_plugin'

    name = 'PyFibre GUI'

    tasks = List(contributes_to=TASKS)

    def _tasks_default(self):
        return [TaskFactory(id='pyfibre.pyfibre_main_task',
                            name='PyFibre GUI (Main)',
                            factory=PyFibreMainTask)
                ]
