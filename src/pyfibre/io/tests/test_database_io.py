import os
from unittest import TestCase
from tempfile import NamedTemporaryFile

import pandas as pd
import numpy as np

from pyfibre.io.database_io import save_database, load_database


class TestDatabaseWriter(TestCase):
    def setUp(self):
        self.data = {
            "one": np.array([1.0, 2.0, 3.0, 4.0]),
            "two": np.array([4.0, 3.0, 2.0, 1.0]),
        }

        self.database = pd.DataFrame(self.data)

    def test_save_database(self):
        with NamedTemporaryFile() as temp_file:
            save_database(self.database, temp_file.name)
            self.assertTrue(os.path.exists(f"{temp_file.name}.h5"))
            self.assertTrue(os.path.exists(f"{temp_file.name}.xlsx"))

            save_database(self.database, temp_file.name, "extra")
            self.assertTrue(os.path.exists(f"{temp_file.name}_extra.h5"))
            self.assertTrue(os.path.exists(f"{temp_file.name}_extra.xlsx"))

    def test_load_database(self):
        with NamedTemporaryFile() as temp_file:
            save_database(self.database, temp_file.name)
            database = load_database(temp_file.name)

            pd.testing.assert_series_equal(self.database["one"], database["one"])
            pd.testing.assert_series_equal(self.database["two"], database["two"])
