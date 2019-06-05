import logging
from unittest import TestCase, mock

import sys, os
import numpy as np

from skimage import data
from scipy.ndimage.filters import gaussian_filter

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = source_dir[:source_dir.rfind(os.path.sep)]
sys.path.append(pyfibre_dir + '/src/')
logger = logging.getLogger(__name__)

THRESH = 1E-7

class TestLogging(TestCase):

	def test_logger(self):

		logger.info('This is a test')

class TestImages(TestCase):

	def setUp(self):

		N = 50
		self.test_images = {}

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

		return self.test_images

	def test_image(self):

		from utilities import get_image_lists, check_analysis
		from preprocessing import load_shg_pl, clip_intensities, nl_means
		from skimage.exposure import equalize_adapthist

		input_files = [pyfibre_dir + '/tests/stubs/test-pyfibre-pl-shg-Stack.tif']
		files, prefixes = get_image_lists(input_files)

		self.assertEqual(len(files[0]), 1)
		self.assertEqual(prefixes[0], pyfibre_dir + '/tests/stubs/test-pyfibre')

		for i, input_file_names in enumerate(files):
			image_path = '/'.join(prefixes[i].split('/')[:-1])
			prefix = prefixes[i]

			image_name = prefix.split('/')[-1]
			# filename = '{}'.format(data_dir + image_name)

			self.assertEqual(image_name, 'test-pyfibre')

			"Load and preprocess image"
			image_shg, image_pl, image_tran = load_shg_pl(input_file_names)

			self.assertEqual(image_shg.shape, (200, 200))
			self.assertEqual(image_pl.shape, (200, 200))
			self.assertEqual(image_tran.shape, (200, 200))

			self.assertAlmostEqual(image_shg.mean(), 0.08748203125, 8)
			self.assertAlmostEqual(image_pl.mean(), 0.1749819688, 8)
			self.assertAlmostEqual(image_tran.mean(), 0.760068620443, 8)

			shg_analysis, pl_analysis = check_analysis(image_shg, image_pl, image_tran)

			self.assertTrue(shg_analysis)
			self.assertTrue(pl_analysis)

			image_shg = clip_intensities(image_shg, p_intensity=(1, 99))
			image_pl = clip_intensities(image_pl, p_intensity=(1, 99))

			self.assertAlmostEqual(image_shg.mean(), 0.17330076923, 8)
			self.assertAlmostEqual(image_pl.mean(), 0.290873620689, 8)

			image_shg = equalize_adapthist(image_shg)

			self.assertAlmostEqual(image_shg.mean(), 0.2386470675, 8)

			image_nl = nl_means(image_shg)

			self.assertAlmostEqual(image_nl.mean(), 0.222907340231, 8)


class TestFunctions(TestCase):

	def setUp(self):
		pass

	def test_string_functions(self):

		from utilities import check_string, check_file_name

		string = "/dir/folder/test_file_SHG.pkl"

		self.assertEqual(check_string(string, -2, '/', 'folder'), "/dir/test_file_SHG.pkl")
		self.assertEqual(check_file_name(string, 'SHG', 'pkl'), "/dir/folder/test_file")


	def test_numeric_functions(self):

		from utilities import unit_vector, numpy_remove, nanmean, ring, matrix_split

		vector = np.array([-3, 2, 6])
		answer = np.array([-0.42857143,  0.28571429,  0.85714286])
		u_vector = unit_vector(vector)

		self.assertAlmostEqual(abs(u_vector - answer).sum(), 0, 7)

		vector_array = np.array([[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]])

		u_vector_array = unit_vector(vector_array)

		self.assertEqual(np.array(vector_array).shape, u_vector_array.shape)

		array_1 = np.arange(50)
		array_2 = array_1 + 20
		answer = np.arange(20)

		edit_array = numpy_remove(array_1, array_2)

		self.assertAlmostEqual(abs(answer - edit_array).sum(), 0, 8)

		array_nan = np.array([2, 3, 1, np.nan])

		self.assertEqual(nanmean(array_nan), 2)

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


class TestFIRE(TestCase):

	def setUp(self):
		pass

	def test_FIRE(self):

		from extraction import (check_2D_arrays, distance_matrix, branch_angles,
							cos_sin_theta_2D)

		pos_2D = np.array([[1, 3],
						   [4, 2],
						   [1, 5]])

		indices = check_2D_arrays(pos_2D, pos_2D + 1.5, 2)

		self.assertEqual(indices[0], 2)
		self.assertEqual(indices[1], 0)

		answer_d_2D = np.array([[[0, 0], [3, -1], [0, 2]],
								[[-3, 1], [0, 0], [-3, 3]],
								[[0, -2], [3, -3], [0, 0]]])
		answer_r2_2D = np.array([[0, 10, 4],
								 [10, 0, 18],
								 [4, 18, 0]])
		d_2D, r2_2D = distance_matrix(pos_2D)

		self.assertAlmostEqual(abs(answer_d_2D - d_2D).sum(), 0, 7)
		self.assertAlmostEqual(abs(answer_r2_2D - r2_2D).sum(), 0, 7)

		direction = np.array([1, 0])
		vectors = d_2D[([2, 0], [0, 1])]
		r = np.sqrt(r2_2D[([2, 0], [0, 1])])

		answer_cos_the = np.array([0, 0.9486833])
		cos_the = branch_angles(direction, vectors, r)
		self.assertAlmostEqual(abs(answer_cos_the - cos_the).sum(), 0, 7)
