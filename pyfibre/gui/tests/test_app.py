from unittest import mock, TestCase

from pyfibre.gui.__main__ import run
from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.tests.utils import delete_log


def mock_pyfibre_constructor(*args, **kwargs):
    mock_pyfibre = mock.Mock(spec=PyFibreGUI)
    mock_pyfibre.run = lambda: None
    return mock_pyfibre


class TestRun(TestCase):

    def setUp(self):
        self.addCleanup(delete_log)

    def test_main(self):

        with mock.patch('pyfibre.gui.__main__.PyFibreGUI') as mock_pyfibre:
            mock_pyfibre.side_effect = mock_pyfibre_constructor

            run(debug=False,
                profile=False)

            self.assertTrue(mock_pyfibre.called)
