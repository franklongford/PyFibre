import os
from unittest import mock, TestCase
import pandas as pd
import numpy as np

from pyfibre.io.database_io import (
    save_database, load_database, check_string,
    check_file_name
)


class TestDatabasewriter(TestCase):

    def setUp(self):

        self.data = {'one': [1., 2., 3., 4.],
                     'two': [4., 3., 2., 1.]}

        self.database = pd.DataFrame(self.data)

    def test_string_functions(self):
        string = "/dir/folder/test_file_SHG.pkl"

        self.assertEqual(
            "/dir/test_file_SHG.pkl",
            check_string(string, -2, '/', 'folder'))
        self.assertEqual(
            "/dir/folder/test_file",
            check_file_name(string, 'SHG', 'pkl'))

    def test_save_database(self):

        save_database(self.database, 'test_database')

        self.assertTrue(os.path.exists('test_database.h5'))
        self.assertTrue(os.path.exists('test_database.xls'))

        save_database(self.database, 'test_database', 'extra')

        self.assertTrue(os.path.exists('test_database_extra.h5'))
        self.assertTrue(os.path.exists('test_database_extra.xls'))

        if os.path.exists('test_database.h5'):
            os.remove('test_database.h5')
        if os.path.exists('test_database.xls'):
            os.remove('test_database.xls')
        if os.path.exists('test_database_extra.h5'):
            os.remove('test_database_extra.h5')
        if os.path.exists('test_database_extra.xls'):
            os.remove('test_database_extra.xls')

    def test_load_database(self):

        save_database(self.database, 'test_database')

        try:
            database = load_database('test_database')

            self.assertAlmostEqual(
                0,
                np.abs(self.database['one'] - database['one']).sum(), 6)
            self.assertAlmostEqual(
                0,
                np.abs(self.database['two'] - database['two']).sum(), 6)

        finally:
            if os.path.exists('test_database.h5'):
                os.remove('test_database.h5')
            if os.path.exists('test_database.xls'):
                os.remove('test_database.xls')