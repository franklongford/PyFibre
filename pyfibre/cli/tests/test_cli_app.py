from unittest import mock, TestCase
from pyfibre.cli.app import parse_files

OS_GETCWD_PATH = 'pyfibre.cli.app.os.getcwd'


class TestCLIApp(TestCase):

    def setUp(self):
        self.name = 'file-shg.tif,/a/path/to/some/file-pl-shg.tif'
        self.directory = ''
        self.key = 'shg'

    def test_parse_files(self):

        with mock.patch(OS_GETCWD_PATH) as mock_getcwd:
            mock_getcwd.return_value = '/a/path/to/some'
            input_files = parse_files(self.name, self.directory, self.key)
            self.assertEqual(input_files[0], '/a/path/to/some/file-shg.tif')
            self.assertEqual(input_files[1], '/a/path/to/some/file-pl-shg.tif')
