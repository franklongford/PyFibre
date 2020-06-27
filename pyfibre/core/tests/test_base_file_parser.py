from unittest import TestCase

from pyfibre.tests.probe_classes.parsers import ProbeFileSet
from pyfibre.tests.fixtures import test_image_path


class TestBaseFileSet(TestCase):

    def setUp(self):
        self.file_set = ProbeFileSet()

    def test_file_set(self):
        self.assertEqual(
            "ProbeFileSet(prefix='/path/to/some/file', "
            "registry={'Probe': '" + test_image_path + "'})",
            repr(self.file_set)
        )
