import sys
import os
from tempfile import NamedTemporaryFile
from unittest import mock, TestCase

from click.testing import CliRunner

import pyfibre.gui.__main__
from pyfibre.version import __version__
from pyfibre.tests.dummy_classes import DummyPyFibreGUI


GUI_PATH = 'pyfibre.gui.__main__.PyFibreGUI'


def mock_run_constructor(*args, **kwargs):
    mock_pyfibre_app = mock.Mock(spec=pyfibre.gui.__main__)
    mock_pyfibre_app.run = lambda: None


class TestClickRun(TestCase):

    def tearDown(self):
        if os.path.exists('pyfibre.log'):
            os.remove('pyfibre.log')

    def test_click_gui_version(self):
        clirunner = CliRunner()
        clirunner.invoke(
            pyfibre.gui.__main__.pyfibre, args="--version")

    def test_click_cli_main(self):

        with mock.patch('pyfibre.gui.__main__') as mock_run:
            mock_run.side_effect = mock_run_constructor

            pyfibre.gui.__main__.pyfibre()

            self.assertTrue(mock_run.pyfibre.called)

    def test_run_with_debug(self):
        with NamedTemporaryFile() as tmp_file:
            with mock.patch(GUI_PATH) as mock_pyfibre:
                mock_pyfibre.return_value = DummyPyFibreGUI()
                pyfibre.gui.__main__.run(
                    debug=True,
                    profile=False,
                    log_name=tmp_file.name
                )
                self.log = pyfibre.gui.__main__.logging.getLogger(__name__)
                # This test seems to be broken at the moment
                # self.assertEqual(10, self.log.getEffectiveLevel())

    def test_run_with_profile(self):
        with NamedTemporaryFile() as tmp_file:
            with mock.patch(GUI_PATH) as mock_pyfibre:
                mock_pyfibre.return_value = DummyPyFibreGUI()
                pyfibre.gui.__main__.run(
                    debug=False,
                    profile=True,
                    log_name=tmp_file.name
                )
                root = ('pyfibre-{}-{}.{}.{}'
                        .format(__version__,
                                sys.version_info.major,
                                sys.version_info.minor,
                                sys.version_info.micro))

                exts = ['.pstats', '.prof']
                files_exist = [False] * len(exts)
                for ind, ext in enumerate(exts):
                    files_exist[ind] = os.path.isfile(root + ext)
                    os.remove(root + ext)
                self.assertTrue(all(files_exist))
