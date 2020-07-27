from tempfile import NamedTemporaryFile
from unittest import mock, TestCase

from pyfibre.gui.__main__ import run
from pyfibre.gui.pyfibre_gui import PyFibreGUI


GUI_PATH = 'pyfibre.gui.__main__.PyFibreGUI'


def mock_pyfibre_constructor(*args, **kwargs):
    mock_pyfibre = mock.Mock(spec=PyFibreGUI)
    mock_pyfibre.run = lambda: None
    return mock_pyfibre


class TestRun(TestCase):

    def test_main(self):

        with NamedTemporaryFile() as tmp_file:
            with mock.patch(GUI_PATH) as mock_pyfibre:
                mock_pyfibre.side_effect = mock_pyfibre_constructor

                run(debug=False,
                    profile=False,
                    log_name=tmp_file.name)

                self.assertTrue(mock_pyfibre.called)
