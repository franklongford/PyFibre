import contextlib
import threading
from unittest import mock, TestCase

from pyface.api import FileDialog, CANCEL
from pyface.tasks.api import TaskWindow
from traits_futures.toolkit_support import toolkit

from pyfibre.tests.dummy_classes import DummyPyFibreGUI
from pyfibre.tests.probe_classes import ProbeTableRow
from pyfibre.gui.pyfibre_main_task import PyFibreMainTask
from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane

GuiTestAssistant = toolkit("gui_test_assistant:GuiTestAssistant")

FILE_DIALOG_PATH = "pyfibre.gui.pyfibre_main_task.FileDialog"
FILE_OPEN_PATH = "pyfibre.io.database_io.save_database"


def mock_run(*args, **kwargs):
    print('mock_run')
    print('done')
    return


def mock_dialog(dialog_class, result, path=''):
    def mock_dialog_call(*args, **kwargs):
        dialog = mock.Mock(spec=dialog_class)
        dialog.open = lambda: result
        dialog.path = path
        return dialog
    return mock_dialog_call


def get_probe_pyfibre_tasks():
    """Returns the Main task, with a mock TaskWindow and dummy
    Application which does not have an event loop."""

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
        super(GuiTestAssistant, self).setUp()
        self.main_task = get_probe_pyfibre_tasks()

    @contextlib.contextmanager
    def long_running_task(self, executor):
        """
        Simulate a long-running task being submitted to the executor.
        The task finishes on exit of the with block.
        """
        event = threading.Event()
        try:
            yield executor.submit_call(event.wait)
        finally:
            event.set()

    def wait_until_stopped(self, executor):
        """"
        Wait for the executor to reach STOPPED state.
        """
        self.run_until(
            executor, "stopped",
            lambda executor: executor.stopped)

    def wait_until_done(self, future):
        self.run_until(
            future, "done",
            lambda future: future.done)

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

    def test_run_cancel_enabled(self):

        self.assertTrue(self.main_task.run_enabled)
        self.assertFalse(self.main_task.stop_enabled)

        with self.long_running_task(
                self.main_task.traits_executor) as future:

            self.main_task.current_futures.append(future)
            self.assertFalse(self.main_task.run_enabled)
            self.assertTrue(self.main_task.stop_enabled)

            self.main_task.traits_executor.stop()

        self.wait_until_stopped(self.main_task.traits_executor)

        self.assertEqual([], self.main_task.current_futures)
        self.assertTrue(self.main_task.run_enabled)
        self.assertFalse(self.main_task.stop_enabled)

    def test_close_saving_dialog(self):
        mock_open = mock.mock_open()
        with mock.patch(FILE_DIALOG_PATH) as mock_file_dialog, mock.patch(
                FILE_OPEN_PATH, mock_open, create=True):
            mock_file_dialog.side_effect = mock_dialog(FileDialog, CANCEL)

            self.main_task.save_database_as()
            mock_open.assert_not_called()

    def test_select_row(self):

        table_row = ProbeTableRow()

        self.main_task.file_display_pane.selected_files = [table_row]
        self.assertEqual(
            table_row,
            self.main_task.viewer_pane.selected_row)
