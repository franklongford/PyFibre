import contextlib
from testfixtures import LogCapture
from unittest import mock, TestCase

import pandas as pd

from pyface.api import FileDialog, CANCEL
from pyface.tasks.api import TaskWindow
from traits_futures.api import MultithreadingContext, submit_call
from traits_futures.toolkit_support import toolkit

from pyfibre.pyfibre_runner import PyFibreRunner
from pyfibre.gui.pyfibre_main_task import PyFibreMainTask, run_analysis
from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane
from pyfibre.tests.dummy_classes import DummyPyFibreGUI
from pyfibre.tests.fixtures import test_image_path
from pyfibre.tests.probe_classes.gui_objects import ProbeTableRow
from pyfibre.tests.probe_classes.factories import ProbeMultiImageFactory
from pyfibre.tests.probe_classes.parsers import ProbeFileSet

GuiTestAssistant = toolkit("gui_test_assistant:GuiTestAssistant")

FILE_DIALOG_PATH = "pyfibre.gui.pyfibre_main_task.FileDialog"
FILE_OPEN_PATH = "pyfibre.io.database_io.save_database"
LOAD_DATABASE = ("pyfibre.tests.probe_classes.analyser.ProbeAnalyser"
                 ".load_databases")
ITERATOR_PATH = 'pyfibre.pyfibre_runner.PyFibreRunner.run'


def dummy_iterate_images(file_sets, analyser, reader):
    for _ in file_sets:
        yield [pd.DataFrame] * len(analyser.database_names)


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

    main_task = PyFibreMainTask(
        multi_image_factories=[ProbeMultiImageFactory()]
    )

    tasks = [main_task]
    mock_window = mock.Mock(spec=TaskWindow)
    mock_window.tasks = tasks
    mock_window.application = pyfibre_gui

    for task in tasks:
        task.window = mock_window
        task.create_central_pane()
        task.create_dock_panes()

    return tasks[0]


class TestPyFibreMainTask(TestCase, GuiTestAssistant):

    def setUp(self):
        GuiTestAssistant.setUp(self)
        self._context = MultithreadingContext()
        self.main_task = get_probe_pyfibre_tasks()
        self.table_row = ProbeTableRow()
        self.file_sets = [ProbeFileSet()]

    def tearDown(self):
        if hasattr(self, "executor"):
            self.executor.stop()
            self.wait_until_stopped(self.executor)
            del self.executor
        self._context.close()
        GuiTestAssistant.tearDown(self)

    @contextlib.contextmanager
    def long_running_task(self, executor):
        """
        Simulate a long-running task being submitted to the executor.
        The task finishes on exit of the with block.
        """
        event = self._context.event()
        try:
            yield submit_call(executor, event.wait)
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
        self.assertIn('Probe', self.main_task.supported_readers)
        self.assertIn('Probe', self.main_task.supported_analysers)

    def test_run_analysis(self):

        with mock.patch(ITERATOR_PATH) as mock_iterate:
            mock_iterate.side_effect = dummy_iterate_images

            run_analysis(
                self.file_sets,
                PyFibreRunner(),
                self.main_task.supported_analysers,
                self.main_task.supported_readers)

            mock_iterate.assert_called_once()

    def test_run_pyfibre(self):

        with mock.patch(ITERATOR_PATH) as mock_iterate:
            mock_iterate.side_effect = dummy_iterate_images

            self.main_task._run_pyfibre()
            self.assertFalse(mock_iterate.called)

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

        self.assertIsNone(self.main_task.viewer_pane.selected_image)
        self.main_task.file_display_pane.selected_files = [self.table_row]
        self.assertIsNotNone(self.main_task.viewer_pane.selected_image)

    def test_create_databases(self):

        def mock_error():
            raise ImportError

        self.main_task.file_display_pane.file_table = [self.table_row]

        with LogCapture() as capture:
            self.main_task.create_databases()
            capture.check(
                ('pyfibre.core.base_multi_image_reader',
                 'INFO',
                 f'Loading {test_image_path}')
            )

            capture.clear()
            with mock.patch(LOAD_DATABASE, side_effect=mock_error):
                self.main_task.create_databases()
                capture.check(
                    ('pyfibre.core.base_multi_image_reader',
                     'INFO',
                     f'Loading {test_image_path}'),
                    ('pyfibre.gui.pyfibre_main_task',
                     'INFO',
                     '/path/to/some/file databases not imported '
                     '- skipping'),
                )

            capture.clear()
            self.main_task.supported_readers = {}
            self.main_task.create_databases()
            capture.check(
                ('pyfibre.gui.pyfibre_main_task',
                 'INFO',
                 '/path/to/some/file analyser / reader not found '
                 '- skipping'),
            )
