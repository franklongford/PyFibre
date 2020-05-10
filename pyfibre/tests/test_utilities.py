from testfixtures import LogCapture
from unittest import mock

import numpy as np
from skimage import data
from scipy.ndimage.filters import gaussian_filter

from pyfibre.io.shg_pl_reader import SHGReader
from pyfibre.utilities import (
    unit_vector, numpy_remove, nanmean, ring, matrix_split,
    label_set, clear_border, flatten_list, log_timer
)

from .probe_classes.utilities import generate_image
from .pyfibre_test_case import PyFibreTestCase


THRESH = 1E-7


class TestImages:

    def __init__(self):

        N = 50
        self.test_images = {}
        self.reader = SHGReader()

        "Make ringed test image"
        image_grid = np.mgrid[:N, :N]
        for i in range(2):
            image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
        image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
        self.test_images['test_image_rings'] = (
                np.sin(10 * np.pi * image_grid / N)
                * np.cos(10 * np.pi * image_grid / N))

        "Make circular test image"
        image_grid = np.mgrid[:N, :N]
        for i in range(2):
            image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
        image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
        self.test_images['test_image_circle'] = (
                1 - gaussian_filter(image_grid, N / 4, 5))

        "Make linear test image"
        test_image = np.zeros((N, N))
        for i in range(4):
            test_image += np.eye(N, N, k=5-i)
        self.test_images['test_image_line'] = test_image

        "Make crossed test image"
        test_image = np.zeros((N, N))
        for i in range(4):
            test_image += np.eye(N, N, k=5-i)
        for i in range(4):
            test_image += np.rot90(np.eye(N, N, k=5-i))
        self.test_images['test_image_cross'] = np.where(test_image != 0, 1, 0)

        "Make noisy test image"
        self.test_images['test_image_noise'] = np.random.random((N, N))

        "Make checkered test image"
        self.test_images['test_image_checker'] = data.checkerboard()


class TestUtilities(PyFibreTestCase):

    def setUp(self):
        (self.image, self.labels,
         self.binary, self.stack) = generate_image()

    def test_unit_vector(self):

        vector = np.array([-3, 2, 6])
        answer = np.array([-0.42857143,  0.28571429,  0.85714286])
        u_vector = unit_vector(vector)

        self.assertArrayAlmostEqual(u_vector, answer, THRESH)

        vector_array = np.array(
            [[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]]
        )

        u_vector_array = unit_vector(vector_array)

        self.assertEqual(
            np.array(vector_array).shape, u_vector_array.shape)

    def test_numpy_remove(self):

        array_1 = np.arange(50)
        array_2 = array_1 + 20
        answer = np.arange(20)

        edit_array = numpy_remove(array_1, array_2)

        self.assertArrayAlmostEqual(answer, edit_array)

        array_nan = np.array([2, 3, 1, np.nan])

        self.assertEqual(nanmean(array_nan), 2)

    def test_label_set(self):

        labels = label_set(self.labels)
        self.assertArrayAlmostEqual(labels, np.array([1, 2]))

        labels = label_set(self.labels, background=-1)
        self.assertArrayAlmostEqual(labels, np.array([0, 1, 2]))

    def test_ring(self):

        ring_answer = np.array([[0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0, 0],
                                [0, 1, 0, 1, 0, 0],
                                [0, 1, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]])

        ring_filter = ring(np.zeros((6, 6)), [2, 2], [1], 1)

        self.assertArrayAlmostEqual(ring_answer, ring_filter)

        split_filter = matrix_split(ring_answer, 2, 2)

        self.assertArrayAlmostEqual(
            split_filter[0],
            np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]))
        self.assertArrayAlmostEqual(
            split_filter[1],
            np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]))
        self.assertArrayAlmostEqual(
            split_filter[2],
            np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]))
        self.assertArrayAlmostEqual(
            split_filter[3],
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))

    def test_clear_border(self):

        clear_answer = np.array([[0, 0, 0, 0, 0, 0],
                                 [0, 1, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 1, 0],
                                 [0, 1, 1, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 0]])

        clear_filter = clear_border(np.ones((6, 6)))
        self.assertArrayAlmostEqual(clear_answer, clear_filter)

        clear_answer = np.array([[0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 1, 0, 0],
                                 [0, 0, 1, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0]])

        clear_filter = clear_border(np.ones((6, 6)), thickness=2)
        self.assertArrayAlmostEqual(clear_answer, clear_filter)

    def test_flatten_list(self):

        list_of_lists = [['0:0', '0:1', '0:2'],
                         ['1:0', '1:1', '1:2'],
                         ['2:0', '2:1', '2:2']]

        flattened_list = flatten_list(list_of_lists)

        self.assertListEqual(
            ['0:0', '0:1', '0:2',
             '1:0', '1:1', '1:2',
             '2:0', '2:1', '2:2'],
            flattened_list
        )

    def test_timer(self):

        @log_timer(message='TEST')
        def function(x, y):
            return x * y

        with LogCapture() as capture:
            with mock.patch('pyfibre.utilities.round',
                            return_value=1.0):
                self.assertEqual(6, function(2, 3))

            capture.check(
                ('pyfibre.utilities',
                 'INFO',
                 'TOTAL TEST TIME = 1.0 s')
            )
