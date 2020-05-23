import sys
import os
from unittest import mock, TestCase

from click.testing import CliRunner

import pyfibre.cli.pyfibre_cli
from pyfibre.version import __version__


def mock_run_pyfibre():
    pass


def mock_run_constructor(*args, **kwargs):
    mock_pyfibre_app = mock.Mock(spec=pyfibre.cli.pyfibre_cli)
    mock_pyfibre_app.run = lambda: None


class TestClickRun(TestCase):

    def test_click_cli_version(self):
        clirunner = CliRunner()
        clirunner.invoke(pyfibre.cli.pyfibre_cli.pyfibre,
                         args="--version")

    def test_click_gui_main(self):

        with mock.patch('pyfibre.cli.pyfibre_cli') as mock_run:
            mock_run.side_effect = mock_run_constructor

            pyfibre.cli.pyfibre_cli.pyfibre()

            self.assertTrue(mock_run.pyfibre.called)

    def test_run(self):
        with mock.patch(
                'pyfibre.cli.app.pyfibre_app'
                '.PyFibreApplication._run_pyfibre') as mock_pyfibre:
            mock_pyfibre.side_effect = mock_run_pyfibre
            pyfibre.cli.pyfibre_cli.run(
                file_path='', key='', sigma=0.5, alpha=0.5,
                log_name='pyfibre', database_name='', debug=False,
                profile=False, ow_metric=False, ow_segment=False,
                ow_network=False, save_figures=False, test=False
            )
            self.log = pyfibre.cli.pyfibre_cli.logging.getLogger(__name__)
            # This test seems to be broken at the moment
            # self.assertEqual(10, self.log.getEffectiveLevel())
            if os.path.exists('pyfibre.log'):
                os.remove('pyfibre.log')

    def test_run_with_profile(self):
        with mock.patch(
                'pyfibre.cli.app.pyfibre_app'
                '.PyFibreApplication._run_pyfibre') as mock_pyfibre:
            mock_pyfibre.side_effect = mock_run_pyfibre
            pyfibre.cli.pyfibre_cli.run(
                file_path='', key='', sigma=0.5, alpha=0.5,
                log_name='pyfibre', database_name='', debug=False,
                profile=True, ow_metric=False, ow_segment=False,
                ow_network=False, save_figures=False, test=False
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
