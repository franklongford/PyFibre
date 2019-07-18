from unittest import mock, TestCase
from pyfibre.cli.app import parse_files

OS_GETCWD_PATH = 'pyfibre.cli.app.os.getcwd'


class TestCLIApp(TestCase):

    def setUp(self):
        self.name = 'file-shg.tif,/a/path/to/some/file-pl-shg.tif'
        self.directory = ''
        self.key = 'shg'
