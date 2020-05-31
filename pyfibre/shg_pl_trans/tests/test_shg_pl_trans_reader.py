from unittest import TestCase

from pyfibre.shg_pl_trans.shg_pl_trans_reader import SHGPLTransReader
from pyfibre.shg_pl_trans.tests.fixtures import (
    directory,
    test_shg_pl_trans_image_path,
    test_shg_image_path,
    test_pl_image_path
)


class TestSHGPLTransReader(TestCase):

    def setUp(self):
        self.reader = SHGPLTransReader()
        self.filenames = [test_shg_pl_trans_image_path,
                          test_shg_image_path,
                          test_pl_image_path]

    def test_load_images(self):

        images = self.reader._load_images(self.filenames[:1])
        self.assertEqual(1, len(images))
        self.assertEqual((3, 200, 200), images[0].shape)

    def test_collate_files(self):
        image_dict = self.reader.collate_files(self.filenames)

        self.assertDictEqual(
            {f'{directory}/test-pyfibre': [test_shg_pl_trans_image_path]},
            image_dict
        )

        image_dict = self.reader.collate_files(self.filenames[1:])

        self.assertDictEqual(
            {f'{directory}/test-pyfibre': [test_shg_image_path,
                                           test_pl_image_path]},
            image_dict
        )

    def test_create_image_stack(self):
        image_stack = self.reader.create_image_stack(
            [test_shg_pl_trans_image_path])
        self.assertEqual(3, len(image_stack))
        self.assertEqual((200, 200), image_stack[0].shape)

        image_stack = self.reader.create_image_stack(
            [test_shg_image_path, test_pl_image_path])
        self.assertEqual(3, len(image_stack))
        self.assertEqual((200, 200), image_stack[0].shape)

    def test_load_multi_image(self):
        multi_image = self.reader.load_multi_image(
            [test_shg_pl_trans_image_path], '/some/path/test-shg-pl-trans')
        self.assertEqual('test-shg-pl-trans', multi_image.name)
        self.assertEqual('/some/path', multi_image.path)
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(3, len(multi_image))

        multi_image = self.reader.load_multi_image(
            [test_shg_image_path, test_pl_image_path],
            'test-shg-pl-trans'
        )
        self.assertEqual('test-shg-pl-trans', multi_image.name)
        self.assertEqual('', multi_image.path)
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(3, len(multi_image))
