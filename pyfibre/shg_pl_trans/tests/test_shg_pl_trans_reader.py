from unittest import TestCase

from pyfibre.shg_pl_trans.shg_pl_trans_reader import SHGPLTransReader
from pyfibre.shg_pl_trans.shg_pl_trans_parser import SHGPLTransFileSet
from pyfibre.shg_pl_trans.tests.fixtures import (
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
        self.file_set = SHGPLTransFileSet(
            prefix='/some/path/test-shg-pl-trans',
            registry={'SHG-PL-Trans': test_shg_pl_trans_image_path,
                      'SHG': test_shg_image_path,
                      'PL-Trans': test_pl_image_path}
        )

    def test_create_image_stack(self):
        image_stack = self.reader.create_image_stack(self.filenames[:1])
        self.assertEqual(3, len(image_stack))
        self.assertEqual((200, 200), image_stack[0].shape)

        image_stack = self.reader.create_image_stack(self.filenames[1:])
        self.assertEqual(3, len(image_stack))
        self.assertEqual((200, 200), image_stack[0].shape)

    def test_load_multi_image(self):
        multi_image = self.reader.load_multi_image(self.file_set)
        self.assertEqual('test-shg-pl-trans', multi_image.name)
        self.assertEqual('/some/path', multi_image.path)
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(3, len(multi_image))

        self.file_set.registry.pop('SHG-PL-Trans')
        multi_image = self.reader.load_multi_image(self.file_set)
        self.assertEqual('test-shg-pl-trans', multi_image.name)
        self.assertEqual('/some/path', multi_image.path)
        self.assertEqual((200, 200), multi_image.shape)
        self.assertEqual(3, len(multi_image))
