from envisage.api import Plugin
from envisage.ui.tasks.api import TaskFactory
from traits.api import List

from pyfibre.ids import TASKS, MULTI_IMAGE_FACTORIES
from pyfibre.gui.pyfibre_main_task import PyFibreMainTask


class PyFibreGUIPlugin(Plugin):
    """ The Plugin containing the PyFibre UI. This contains the
    factories which create the Tasks (currently Main)"""

    id = 'pyfibre.core.pyfibre_gui_plugin'

    name = 'PyFibre GUI'

    tasks = List(contributes_to=TASKS)

    def _tasks_default(self):
        return [TaskFactory(id='pyfibre.pyfibre_main_task',
                            name='PyFibre GUI (Main)',
                            factory=self._create_main_task)
                ]

    def _create_main_task(self):

        factories = self.application.get_extensions(
            MULTI_IMAGE_FACTORIES)

        pyfibre_main_task = PyFibreMainTask(
            multi_image_factories=factories,
        )

        return pyfibre_main_task
