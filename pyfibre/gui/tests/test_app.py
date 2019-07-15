from unittest import mock, TestCase

from pyfibre.gui.app import run
from pyfibre.gui.pyfibre_gui import PyFibreGUI


def mock_pyfibre_constructor(*args, **kwargs):
    mock_pyfibre = mock.Mock(spec=PyFibreGUI)
    mock_pyfibre.run = lambda: None
    return mock_pyfibre


class TestRun(TestCase):

    def test_main(self):
        with mock.patch('pyfibre.gui.app.PyFibreGUI') as mock_pyfibre:
            mock_pyfibre.side_effect = mock_pyfibre_constructor

            run(debug=False,
                profile=False,
                window_size=(1680, 1050))

            self.assertTrue(mock_pyfibre.called)
