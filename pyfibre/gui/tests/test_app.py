from unittest import mock, TestCase
import os

from pyfibre.gui.__main__ import run
from pyfibre.gui.pyfibre_gui import PyFibreGUI


def mock_pyfibre_constructor(*args, **kwargs):
    mock_pyfibre = mock.Mock(spec=PyFibreGUI)
    mock_pyfibre.run = lambda: None
    return mock_pyfibre


class TestRun(TestCase):

    def tearDown(self):
        if os.path.exists('pyfibre.log'):
            os.remove('pyfibre.log')

    def test_main(self):

        with mock.patch('pyfibre.gui.__main__.PyFibreGUI') as mock_pyfibre:
            mock_pyfibre.side_effect = mock_pyfibre_constructor

            run(debug=False,
                profile=False)

            self.assertTrue(mock_pyfibre.called)
