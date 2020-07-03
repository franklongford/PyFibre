from unittest import TestCase

from pyfibre.tests.probe_classes.parsers import ProbeFileSet


class TestBaseFileSet(TestCase):

    def setUp(self):
        self.file_set = ProbeFileSet()

    def test_file_set(self):
        self.assertEqual(
            "ProbeFileSet(prefix='/path/to/some/file')",
            repr(self.file_set)
        )
