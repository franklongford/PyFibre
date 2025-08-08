import numpy as np

from pyfibre.tests.probe_classes.objects import ProbeSegment
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase
from pyfibre.utilities import NotSupportedError


class TestBaseSegment(PyFibreTestCase):

    def setUp(self):
        self.array = np.zeros((6, 8))
        self.array[:, 4] = 1
        self.array[2, 4:] = 1
        self.segment = ProbeSegment()

    def test_not_implemented(self):

        with self.assertRaises(NotSupportedError):
            self.segment.to_json()

        with self.assertRaises(NotSupportedError):
            ProbeSegment.from_json(None)

    def test_to_array(self):
        array = self.segment.to_array()
        self.assertArrayAlmostEqual(
            self.array, array
        )

    def test_from_array(self):
        array = self.segment.to_array(shape=(6, 8))
        new_segment = ProbeSegment.from_array(array)

        self.assertEqual(
            self.segment.region.bbox,
            new_segment.region.bbox
        )

    def test_tags(self):

        self.assertEqual(
            'Test Segment', self.segment._shape_tag)

    def test_generate_database(self):

        shape_metrics = ['Area', 'Circularity', 'Eccentricity',
                         'Coverage']
        texture_metrics = ['Mean', 'STD', 'Entropy']

        database = self.segment.generate_database()
        self.assertEqual(7, len(database))

        for metric in shape_metrics + texture_metrics:
            self.assertIn(f'Test Segment {metric}', database)

        database = self.segment.generate_database(
            image_tag='Test')
        self.assertEqual(7, len(database))

        for metric in shape_metrics:
            self.assertIn(f'Test Segment {metric}', database)

        for metric in texture_metrics:
            self.assertIn(f'Test Segment Test {metric}', database)

        self.segment.region = None
        with self.assertRaises(AttributeError):
            self.segment.generate_database()
