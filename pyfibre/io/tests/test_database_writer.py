from unittest import mock, TestCase

from pyfibre.io.database_writer import (
    write_database, check_string, check_file_name
)


class TestDatabasewriter(TestCase):

    def setUp(self):

        pass

    def test_string_functions(self):
        string = "/dir/folder/test_file_SHG.pkl"

        self.assertEqual(check_string(string, -2, '/', 'folder'), "/dir/test_file_SHG.pkl")
        self.assertEqual(check_file_name(string, 'SHG', 'pkl'), "/dir/folder/test_file")

