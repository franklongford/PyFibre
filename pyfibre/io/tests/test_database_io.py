import os
from unittest import TestCase
from tempfile import NamedTemporaryFile

import pandas as pd
import numpy as np

from pyfibre.io.database_io import (
    save_database, load_database
)


class TestDatabaseWriter(TestCase):

    def setUp(self):

        self.data = {'one': np.array([1., 2., 3., 4.]),
                     'two': np.array([4., 3., 2., 1.])}

        self.database = pd.DataFrame(self.data)

    def test_save_database(self):

        with NamedTemporaryFile() as temp_file:

            save_database(self.database, temp_file.name)
            self.assertTrue(os.path.exists(f'{temp_file.name}.h5'))
            self.assertTrue(os.path.exists(f'{temp_file.name}.xls'))

            save_database(self.database, temp_file.name, 'extra')
            self.assertTrue(os.path.exists(f'{temp_file.name}_extra.h5'))
            self.assertTrue(os.path.exists(f'{temp_file.name}_extra.xls'))

    def test_load_database(self):

        with NamedTemporaryFile() as temp_file:

            save_database(self.database, temp_file.name)
            database = load_database(temp_file.name)

            self.assertAlmostEqual(
                0,
                np.abs(self.database['one'] - database['one']).sum(), 6)
            self.assertAlmostEqual(
                0,
                np.abs(self.database['two'] - database['two']).sum(), 6)