from unittest import mock, TestCase

from envisage.api import Application

from pyfibre.gui.pyfibre_plugin import PyFibrePlugin

MAIN_TASK = ("pyfibre.gui.pyfibre_main_task"
             ".PyFibreMainTask")
PLUGIN_SERVICE = 'envisage.api.Plugin.application.get_service'


def mock_return_none(*args, **kwargs):
    return


class TestWfManagerPlugin(TestCase):

    def setUp(self):
        self.pyfibre_plugin = PyFibrePlugin()
        self.pyfibre_plugin.application = mock.Mock(spec=Application)

    def test_init(self):
        self.assertEqual(1, len(self.pyfibre_plugin.tasks))
        self.assertEqual(
            "PyFibre GUI (Main)",
            self.pyfibre_plugin.tasks[0].name,
        )
