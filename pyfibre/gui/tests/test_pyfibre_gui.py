import contextlib

from unittest import TestCase

from pyface.ui.qt4.util.gui_test_assistant import GuiTestAssistant

from pyfibre.tests.probe_classes import ProbePyFibreGUI


class TestPyFibreGUI(GuiTestAssistant, TestCase):

    def setUp(self):
        super(TestPyFibreGUI, self).setUp()
        self.pyfibre_gui = ProbePyFibreGUI()

    @contextlib.contextmanager
    def create_tasks(self):
        self.pyfibre_gui.run()
        self.main_task = self.pyfibre_gui.windows[0].tasks[0]
        try:
            yield
        finally:
            for plugin in self.pyfibre_gui:
                self.pyfibre_gui.remove_plugin(plugin)
            self.pyfibre_gui.exit()

    def test_init(self):

        """
        with self.create_tasks():
            self.assertEqual(1, len(self.pyfibre_gui.default_layout))
            self.assertEqual(1, len(self.pyfibre_gui.task_factories))
            self.assertEqual(1, len(self.pyfibre_gui.windows))
            self.assertEqual(1, len(self.pyfibre_gui.windows[0].tasks))

            self.assertIsInstance(
                self.main_task, PyFibreMainTask)
        """
