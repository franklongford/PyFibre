from unittest import TestCase

from pyfibre.io.shg_pl_reader import (
    get_image_type, extract_prefix,
    get_files_prefixes, filter_input_files,
    populate_image_dictionary,
    collate_image_dictionary, SHGReader,
    SHGPLTransReader
)
from pyfibre.tests.fixtures import (
    test_shg_image_path, test_pl_image_path,
    test_shg_pl_trans_image_path)


class TestImageReader(TestCase):

    def test_get_image_type(self):
        self.assertEqual(
            'SHG-PL-Trans', get_image_type('some-pl-shg-test.tif'))
        self.assertEqual(
            'PL-Trans', get_image_type('some-pl-test.tif'))
        self.assertEqual(
            'SHG', get_image_type('some-shg-test.tif'))

        # Test failure
        self.assertEqual(
            'Unknown', get_image_type('some-psh-test.tif'))

    def test_extract_prefix(self):
        self.assertEqual(
            '/directory/prefix',
            extract_prefix(
                '/directory/prefix-pl-shg-test.tif', '-pl-shg'))
        self.assertEqual(
            '/directory/prefix',
            extract_prefix(
                '/directory/prefix-pl-test.tif', '-pl'))
        self.assertEqual(
            '/directory/prefix',
            extract_prefix(
                '/directory/prefix-shg-test.tif', '-shg'))

    def test_get_files_prefixes(self):
        input_files = ['/directory/prefix1-pl-shg-test.tif',
                       '/directory/prefix2-pl-shg-test.tif',
                       '/directory/prefix-shg-test.tif']

        files, prefixes = get_files_prefixes(input_files, '-pl-shg')
        self.assertListEqual(
            ['/directory/prefix1-pl-shg-test.tif',
             '/directory/prefix2-pl-shg-test.tif'], files)
        self.assertListEqual(
            ['/directory/prefix1', '/directory/prefix2'], prefixes)

    def test_filter_input_files(self):
        input_files = ['/directory/prefix1-pl-shg-test.png',
                       '/directory/prefix2-pl-shg-test.tif',
                       '/directory/prefix-display-test.tif',
                       '/directory/prefix-shg-virada.tif',
                       '/directory/prefix-shg-asterisco.tif']

        filtered_files = filter_input_files(input_files)

        self.assertListEqual(
            ['/directory/prefix2-pl-shg-test.tif'], filtered_files)

    def test_populate_image_dictionary(self):
        input_files = ['/directory/prefix-pl-shg-test.tif',
                       '/directory/prefix-pl-test.tif',
                       '/directory/prefix-shg-test.tif']
        image_dict = {}

        populate_image_dictionary(
            input_files, image_dict, 'SHG-PL-Trans', 'pl-shg')

        self.assertDictEqual(
            {'/directory/prefix': {
                'SHG-PL-Trans': '/directory/prefix-pl-shg-test.tif'}},
            image_dict)
        self.assertListEqual(
            ['/directory/prefix-pl-test.tif',
             '/directory/prefix-shg-test.tif'],
            input_files)

        populate_image_dictionary(
            input_files, image_dict, 'PL-Trans', 'pl')

        self.assertDictEqual(
            {
                '/directory/prefix':
                    {'SHG-PL-Trans': '/directory/prefix-pl-shg-test.tif',
                     'PL-Trans': '/directory/prefix-pl-test.tif'}
            },
            image_dict)
        self.assertListEqual(
            ['/directory/prefix-shg-test.tif'],
            input_files)

    def test_collate_image_dictionary(self):
        input_files = ['/directory/prefix-pl-shg-test.tif',
                       '/directory/prefix-pl-test.tif',
                       '/directory/prefix-shg-test.tif',
                       '/directory/prefix-pl-display.tif']

        image_dict = collate_image_dictionary(input_files)

        self.assertDictEqual(
            {'/directory/prefix': {
                'SHG-PL-Trans': '/directory/prefix-pl-shg-test.tif',
                'PL-Trans': '/directory/prefix-pl-test.tif',
                'SHG': '/directory/prefix-shg-test.tif'}},
            image_dict)
        self.assertEqual(4, len(input_files))


class TestSHGReader(TestCase):

    def setUp(self):
        self.reader = SHGReader()
        self.filenames = [test_shg_image_path]

    def test_create_image_stack(self):

        image_stack = self.reader.create_image_stack(self.filenames)
        self.assertEqual(1, len(image_stack))
        self.assertEqual((200, 200), image_stack[0].shape)

    def test_load_multi_image(self):

        multi_image = self.reader.load_multi_image(self.filenames)
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(1, len(multi_image))


class TestSHGPLTransReader(TestCase):

    def setUp(self):
        self.reader = SHGPLTransReader()
        self.filenames = [test_shg_pl_trans_image_path]

    def test_create_image_stack(self):
        image_stack = self.reader.create_image_stack(self.filenames)
        self.assertEqual(3, len(image_stack))
        self.assertEqual((200, 200), image_stack[0].shape)

        image_stack = self.reader.create_image_stack(
            [test_shg_image_path, test_pl_image_path])
        self.assertEqual(3, len(image_stack))
        self.assertEqual((200, 200), image_stack[0].shape)

    def test_load_multi_image(self):
        multi_image = self.reader.load_multi_image(self.filenames)
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(3, len(multi_image))

        multi_image = self.reader.load_multi_image(
            [test_shg_image_path, test_pl_image_path])
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(3, len(multi_image))
