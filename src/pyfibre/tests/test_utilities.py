from unittest import mock, TestCase

import numpy as np

from pyfibre.utilities import (
    unit_vector,
    numpy_remove,
    nanmean,
    ring,
    matrix_split,
    label_set,
    clear_border,
    flatten_list,
    log_time,
)
from pyfibre.testing.example_objects import generate_image


THRESH = 1e-7


class TestUtilities(TestCase):
    def setUp(self):
        (self.image, self.labels, self.binary, self.stack) = generate_image()

    def test_unit_vector(self):
        vector = np.array([-3, 2, 6])
        answer = np.array([-0.42857143, 0.28571429, 0.85714286])
        u_vector = unit_vector(vector)

        np.testing.assert_almost_equal(u_vector, answer, THRESH)

        vector_array = np.array([[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]])

        u_vector_array = unit_vector(vector_array)

        self.assertEqual(np.array(vector_array).shape, u_vector_array.shape)

    def test_numpy_remove(self):
        array_1 = np.arange(50)
        array_2 = array_1 + 20
        answer = np.arange(20)

        edit_array = numpy_remove(array_1, array_2)

        np.testing.assert_almost_equal(answer, edit_array)

    def test_nanmean(self):
        array_nan = np.array([2, 3, 1, np.nan])
        self.assertEqual(nanmean(array_nan), 2)

        weights = np.array([3, 2, 5, 1])
        self.assertAlmostEqual(nanmean(array_nan, weights), 1.7)

        array_nan = np.array([2, 3, 1, None])
        self.assertEqual(nanmean(array_nan), 2)

    def test_label_set(self):
        labels = label_set(self.labels)
        np.testing.assert_almost_equal(labels, np.array([1, 2]))

        labels = label_set(self.labels, background=-1)
        np.testing.assert_almost_equal(labels, np.array([0, 1, 2]))

    def test_ring(self):
        ring_answer = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        ring_filter = ring(np.zeros((6, 6)), [2, 2], [1], 1)

        np.testing.assert_almost_equal(ring_answer, ring_filter)

        split_filter = matrix_split(ring_answer, 2, 2)

        np.testing.assert_almost_equal(
            split_filter[0], np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
        )
        np.testing.assert_almost_equal(
            split_filter[1], np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
        )
        np.testing.assert_almost_equal(
            split_filter[2], np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        )
        np.testing.assert_almost_equal(
            split_filter[3], np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        )

    def test_clear_border(self):
        clear_answer = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        clear_filter = clear_border(np.ones((6, 6)))
        np.testing.assert_almost_equal(clear_answer, clear_filter)

        clear_answer = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        clear_filter = clear_border(np.ones((6, 6)), thickness=2)
        np.testing.assert_almost_equal(clear_answer, clear_filter)

    def test_flatten_list(self):
        list_of_lists = [
            ["0:0", "0:1", "0:2"],
            ["1:0", "1:1", "1:2"],
            ["2:0", "2:1", "2:2"],
        ]

        flattened_list = flatten_list(list_of_lists)

        self.assertListEqual(
            ["0:0", "0:1", "0:2", "1:0", "1:1", "1:2", "2:0", "2:1", "2:2"],
            flattened_list,
        )

    def test_timer(self):
        @log_time(message="TEST")
        def function(x, y):
            return x * y

        with self.assertLogs("pyfibre.utilities") as capture:
            with mock.patch("pyfibre.utilities.round", return_value=1.0):
                self.assertEqual(6, function(2, 3))

            self.assertIn(
                "INFO:pyfibre.utilities:TOTAL TEST TIME = 1.0 s", capture.output
            )
