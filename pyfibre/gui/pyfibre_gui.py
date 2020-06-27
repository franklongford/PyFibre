import logging
import os

from envisage.ui.tasks.api import TasksApplication
from pyface.image_resource import ImageResource
from pyface.tasks.api import TaskWindowLayout
from pyface.api import SplashScreen
from traits.api import (
    Tuple, Int, List, Bool
)


BACKGROUND_COLOUR = '#d8baa9'
logger = logging.getLogger(__name__)


class PyFibreGUI(TasksApplication):

    id = 'pyfibre.pyfibre_gui'

    name = 'PyFibre GUI'

    window_size = Tuple((1680, 1050))

    splash_screen = SplashScreen(image=ImageResource("images/splash"))

    # The default window-level layout for the application.
    default_layout = List(TaskWindowLayout)

    # Whether to restore the previous application-level layout
    # when the application is started.
    always_use_default_layout = Bool(True)

    n_proc = Int(1)

    def _default_layout_default(self):
        tasks = [factory.id for factory in self.task_factories]
        return [TaskWindowLayout(
            *tasks,
            active_task='pyfibre.pyfibre_main_task',
            size=self.window_size
        )]

    def _load_state(self):
        super(PyFibreGUI, self)._load_state()
        if (
                self._state.window_layouts
                and self._state.window_layouts[0].get_active_task() is None
        ):
            # This is a possible way a corrupted state file would manifest
            # Remove it and try again with a default state.
            state_file = os.path.join(
                self.state_location, 'application_memento')
            if os.path.exists(state_file):
                os.unlink(state_file)
                logger.warning("The state file at {!r} was corrupted and has "
                               "been removed.".format(state_file))
            super(PyFibreGUI, self)._load_state()
