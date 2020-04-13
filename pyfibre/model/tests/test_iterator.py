from unittest import TestCase

from pyfibre.model.iterator import assign_images


class TestIterator(TestCase):

    def setUp(self):

        self.image_dictionary = {
            'SHG-PL-Trans': '/directory/prefix-pl-shg-test.tif',
            'PL-Trans': '/directory/prefix-pl-test.tif',
            'SHG': '/directory/prefix-shg-test.tif'}

    def test_assign_images(self):

        filenames, image_type = assign_images(self.image_dictionary)
        self.assertEqual(1, len(filenames))
        self.assertEqual('SHG-PL-Trans', image_type)
        self.assertEqual(
            '/directory/prefix-pl-shg-test.tif',
            filenames[0])

        self.image_dictionary.pop('SHG-PL-Trans')
        filenames, image_type = assign_images(self.image_dictionary)
        self.assertEqual(2, len(filenames))
        self.assertEqual('SHG-PL-Trans', image_type)
        self.assertEqual(
            '/directory/prefix-shg-test.tif',
            filenames[0])
        self.assertEqual(
            '/directory/prefix-pl-test.tif',
            filenames[1])
