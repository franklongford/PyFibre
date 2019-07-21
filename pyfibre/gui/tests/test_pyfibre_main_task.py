from unittest import mock, TestCase

from pyface.ui.qt4.util.gui_test_assistant import GuiTestAssistant
from pyface.tasks.api import TaskWindow

from traits.api import Bool

from pyfibre.gui.tests.utils import DummyPyFibreGUI
from pyfibre.gui.pyfibre_main_task import PyFibreMainTask

from pyfibre.gui.title_pane import TitlePane
from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane


def get_probe_pyfibre_tasks():
    # Returns the Main task, with a mock TaskWindow and dummy
    # Application which does not have an event loop.

    pyfibre_gui = DummyPyFibreGUI()

    main_task = PyFibreMainTask()

    tasks = [main_task]
    mock_window = mock.Mock(spec=TaskWindow)
    mock_window.tasks = tasks
    mock_window.application = pyfibre_gui

    for task in tasks:
        task.window = mock_window
        task.create_central_pane()
        task.create_dock_panes()

    return tasks[0]


class TestPyFibreMainTask(GuiTestAssistant, TestCase):

    def setUp(self):
        super(TestPyFibreMainTask, self).setUp()
        self.main_task = get_probe_pyfibre_tasks()

    def test___init__(self):
        self.assertIsInstance(
            self.main_task.create_central_pane(), ViewerPane
        )
        self.assertIsInstance(
            self.main_task.create_dock_panes()[0], FileDisplayPane
        )
        self.assertIsInstance(
            self.main_task.create_dock_panes()[1], OptionsPane
        )

    def test_open_file(self):

        print(self.main_task.window.__dict__.keys())