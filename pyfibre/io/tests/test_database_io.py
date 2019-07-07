from unittest import mock, TestCase

from pyfibre.io.database_io import (
    save_database, check_string, check_file_name
)


class TestDatabasewriter(TestCase):

    def setUp(self):

        pass

    def test_string_functions(self):
        string = "/dir/folder/test_file_SHG.pkl"

        self.assertEqual(
            "/dir/test_file_SHG.pkl",
            check_string(string, -2, '/', 'folder'))
        self.assertEqual(
            "/dir/folder/test_file",
            check_file_name(string, 'SHG', 'pkl'))
