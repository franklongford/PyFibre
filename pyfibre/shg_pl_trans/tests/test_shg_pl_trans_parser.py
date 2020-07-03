from unittest import TestCase

from pyfibre.shg_pl_trans.shg_pl_trans_parser import SHGPLTransParser


class TestSHGPLTransParser(TestCase):

    def setUp(self):
        self.parser = SHGPLTransParser()
        self.input_files = ['not/a/file.tif',
                            'a/file-shg.tif',
                            'a/file-pl.tif',
                            'a/full_file-pl-shg.tif']

    def test_cache_file_sets(self):

        self.parser._cache_file_sets(self.input_files, 'SHG')
        self.assertIn('a/file', self.parser._file_set_cache)
        self.assertEqual(1, len(self.parser._file_set_cache))

        self.parser._cache_file_sets(self.input_files, 'PL-Trans')
        self.assertIn('a/file', self.parser._file_set_cache)
        self.assertEqual(1, len(self.parser._file_set_cache))

        self.parser._cache_file_sets(self.input_files, 'SHG-PL-Trans')
        self.assertIn('a/full_file', self.parser._file_set_cache)
        self.assertEqual(2, len(self.parser._file_set_cache))

    def test_get_file_sets(self):

        file_sets = self.parser.get_file_sets(self.input_files)
        self.assertEqual(2, len(file_sets))

        self.assertIn('SHG-PL-Trans', file_sets[0].registry)
        self.assertIn('SHG', file_sets[1].registry)
        self.assertIn('PL-Trans', file_sets[1].registry)
