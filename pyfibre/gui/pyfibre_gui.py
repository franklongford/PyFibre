import logging
import os

from envisage.ui.tasks.api import (
    TasksApplication
)

from pyface.tasks.api import TaskWindowLayout

from traits.api import (
    Either, Tuple, Int, List, Property, Bool, Supports
)

BACKGROUND_COLOUR = '#d8baa9'
logger = logging.getLogger(__name__)


class PyFibreGUI(TasksApplication):

    id = 'pyfibre.pyfibre_gui'

    name = 'PyFibre GUI'

    window_size = Tuple((1680, 1050))

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

    # FIXME: This isn't needed if the bug in traitsui/qt4/ui_panel.py is fixed
    def _application_exiting_fired(self):
        self._remove_tasks()

    def _remove_tasks(self):
        """Removes the task elements from all windows in the application.
        Part of a workaround for a bug in traitsui/qt4/ui_panel.py where
        sizeHint() would be called, even when a Widget was already destroyed"""
        for window in self.windows:
            tasks = window.tasks
            for task in tasks:
                window.remove_task(task)

    # FIXME: If the underlying envisage TasksApplication function is fixed to
    #        work correctly, this will not be needed.
    def create_window(self, layout, restore, **traits):
        """ Creates a new TaskWindow.
        """
        window = super(PyFibreGUI, self).create_window(
            layout, not restore, **traits
        )
        return window

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
