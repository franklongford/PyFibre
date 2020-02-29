import os
import contextlib
import threading
import time
from unittest import mock, TestCase

from pyface.tasks.api import TaskWindow

from traits.api import Bool
from traits_futures.api import CallFuture
from traits_futures.toolkit_support import toolkit

from pyfibre.gui.tests.utils import DummyPyFibreGUI
from pyfibre.gui.pyfibre_main_task import PyFibreMainTask

from pyfibre.gui.title_pane import TitlePane
from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane

GuiTestAssistant = toolkit("gui_test_assistant:GuiTestAssistant")

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
        self.run_until(executor, "stopped", lambda executor: executor.stopped)

    def wait_until_done(self, future):
        self.run_until(future, "done", lambda future: future.done)

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

    def test__run_pyfibre(self):
        test_file_path = (
            pyfibre_dir + '/tests/fixtures/test-pyfibre-pl-shg-Stack.tif'
        )

        self.main_task.file_display_pane.add_files(test_file_path)

        with mock.patch(
                'pyfibre.gui.pyfibre_main_task.PyFibreMainTask.create_databases'
            ) as mock_create_databases:
            mock_create_databases.side_effect = mock_run

            self.assertTrue(self.main_task.run_enabled)
            self.assertFalse(self.main_task.stop_enabled)

            self.main_task._run_pyfibre()

            future = self.main_task.current_futures[0]

        self.wait_until_done(future)

        self.assertTrue(self.main_task.run_enabled)
        self.assertFalse(self.main_task.stop_enabled)
        self.assertEqual([], self.main_task.current_futures)
