import logging
from unittest import TestCase

import sys, os
import numpy as np

from skimage import data
from scipy.ndimage.filters import gaussian_filter

from pyfibre.io.tif_reader import TIFReader
from pyfibre.utilities import unit_vector, numpy_remove, nanmean, ring, matrix_split


source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = source_dir[:source_dir.rfind(os.path.sep)]
sys.path.append(pyfibre_dir + '/pyfibre/')
logger = logging.getLogger(__name__)

THRESH = 1E-7


class TestLogging(TestCase):

	def test_logger(self):

		logger.info('This is a test')


class TestImages(TestCase):

	def setUp(self):

		N = 50
		self.test_images = {}
		self.reader = TIFReader()

		"Make ringed test image"
		image_grid = np.mgrid[:N, :N]
		for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
		image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
		self.test_images['test_image_rings'] = np.sin(10 * np.pi * image_grid / N ) * np.cos(10 * np.pi * image_grid / N)

		"Make circular test image"
		image_grid = np.mgrid[:N, :N]
		for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
		image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
		self.test_images['test_image_circle'] = 1 - gaussian_filter(image_grid, N / 4, 5)

		"Make linear test image"
		test_image = np.zeros((N, N))
		for i in range(4): test_image += np.eye(N, N, k=5-i)
		self.test_images['test_image_line'] = test_image

		"Make crossed test image"
		test_image = np.zeros((N, N))
		for i in range(4): test_image += np.eye(N, N, k=5-i)
		for i in range(4): test_image += np.rot90(np.eye(N, N, k=5-i))
		self.test_images['test_image_cross'] = np.where(test_image != 0, 1, 0)

		"Make noisy test image"
		self.test_images['test_image_noise'] = np.random.random((N, N))

		"Make checkered test image"
		self.test_images['test_image_checker'] = data.checkerboard()

	def test_image(self):

		input_files = [pyfibre_dir + '/tests/stubs/test-pyfibre-pl-shg-Stack.tif']
		reader = TIFReader(input_files)
		reader.load_multi_images()

		for prefix, data in reader.files.items():
			multi_image = data['image']
			image_name = os.path.basename(prefix)

			self.assertEqual(image_name, 'test-pyfibre')

			self.assertEqual(multi_image.image_shg.shape, (200, 200))
			self.assertEqual(multi_image.image_pl.shape, (200, 200))
			self.assertEqual(multi_image.image_tran.shape, (200, 200))
			self.assertTrue(multi_image.shg_analysis)
			self.assertTrue(multi_image.pl_analysis)


class TestUtilities(TestCase):

	def setUp(self):
		pass

	def test_unit_vector(self):

		vector = np.array([-3, 2, 6])
		answer = np.array([-0.42857143,  0.28571429,  0.85714286])
		u_vector = unit_vector(vector)

		self.assertAlmostEqual(abs(u_vector - answer).sum(), 0, 7)

		vector_array = np.array([[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]])

		u_vector_array = unit_vector(vector_array)

		self.assertEqual(np.array(vector_array).shape, u_vector_array.shape)

	def test_numpy_remove(self):

		array_1 = np.arange(50)
		array_2 = array_1 + 20
		answer = np.arange(20)

		edit_array = numpy_remove(array_1, array_2)

		self.assertAlmostEqual(abs(answer - edit_array).sum(), 0, 8)

		array_nan = np.array([2, 3, 1, np.nan])

		self.assertEqual(nanmean(array_nan), 2)

	def test_ring(self):

		ring_answer = np.array([[0, 0, 0, 0, 0, 0],
								[0, 1, 1, 1, 0, 0],
								[0, 1, 0, 1, 0, 0],
								[0, 1, 1, 1, 0, 0],
								[0, 0, 0, 0, 0, 0],
								[0, 0, 0, 0, 0, 0]])

		ring_filter = ring(np.zeros((6, 6)), [2, 2], [1], 1)

		self.assertAlmostEqual(abs(ring_answer - ring_filter).sum(), 0, 8)

		split_filter = matrix_split(ring_answer, 2, 2)

		self.assertAlmostEqual(abs(
			split_filter[0] - np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])).sum(), 0, 8)
		self.assertAlmostEqual(abs(
			split_filter[1] - np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])).sum(), 0, 8)
		self.assertAlmostEqual(abs(
			split_filter[2] - np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])).sum(), 0, 8)
		self.assertAlmostEqual(abs(
			split_filter[3] - np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])).sum(), 0, 8)
