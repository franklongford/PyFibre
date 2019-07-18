import logging
from multiprocessing import Queue

from pyface.tasks.action.api import (
    SMenu, SMenuBar, SToolBar, TaskAction, TaskToggleGroup
)
from pyface.tasks.api import (
    PaneItem, Task, TaskLayout
)

from traits.api import (
    Bool, Int, List, Float, Instance
)

from pyfibre.gui.title_pane import TitlePane
from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane

logger = logging.getLogger(__name__)


class PyFibreMainTask(Task):

    id = 'pyfibre.pyfibre_main_task'

    name = 'PyFibre GUI (Main)'

    title_pane = Instance(TitlePane)

    options_pane = Instance(OptionsPane)

    file_display_pane = Instance(FileDisplayPane)

    # Multiprocessor list
    processes = Instance(List)

    #: The menu bar for this task.
    menu_bar = Instance(SMenuBar)

    def _default_layout_default(self):
        """ Defines the default layout of the task window """
        return TaskLayout(
            top=PaneItem('pyfibre.title_pane'),
            left=PaneItem('pyfibre.options_pane'),
            right=PaneItem('pyfibre.file_display_pane')
        )

    def _menu_bar_default(self):
        """A menu bar with functions relevant to the Setup task.
        """
        menu_bar = SMenuBar(SMenu(id='File', name='&File'),
                            SMenu(id='Edit', name='&Edit'),
                            SMenu(TaskToggleGroup(),
                                  id='View', name='&View'))

        return menu_bar

    def create_central_pane(self):
        """ Creates the central pane
        """
        return ViewerPane()

    def create_dock_panes(self):
        """ Creates the dock panes
        """
        return [self.title_pane,
                self.file_display_pane,
                self.options_pane]

    def _title_pane_default(self):
        return TitlePane()

    def _options_pane_default(self):
        return OptionsPane()

    def _file_display_pane_default(self):
        return FileDisplayPane()

