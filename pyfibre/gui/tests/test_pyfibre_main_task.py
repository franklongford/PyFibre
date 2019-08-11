import os
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

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = os.path.dirname(os.path.dirname(source_dir))


def mock_run(*args, **kwargs):
    print('mock_run')
    print('done')
    return


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

    def test__run_pyfibre(self):
        test_file_path = (
            pyfibre_dir + '/tests/fixtures/test-pyfibre-pl-shg-Stack.tif'
        )

        self.main_task.file_display_pane.add_files(test_file_path)

        with mock.patch(
                'pyfibre.gui.pyfibre_main_task.process_run'
            ) as mock_process_run, \
                mock.patch(
                'pyfibre.gui.pyfibre_main_task.PyFibreMainTask.create_databases'
            ) as mock_create_databases, \
                mock.patch(
                'pyfibre.gui.pyfibre_main_task.PyFibreMainTask.stop_run'
            ) as mock_stop_run:

            mock_process_run.side_effect = mock_run
            mock_create_databases.side_effect = mock_run

            self.main_task._run_pyfibre()
