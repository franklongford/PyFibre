import sys, os
import numpy as np

from skimage import data
from scipy.ndimage.filters import gaussian_filter

source_dir = os.path.dirname(os.path.realpath(__file__))
pyfibre_dir = source_dir[:source_dir.rfind(os.path.sep)]
sys.path.append(pyfibre_dir + '/src/')

THRESH = 1E-7

def create_images(N=50):

	test_images = {}

	"Make ringed test image"
	image_grid = np.mgrid[:N, :N]
	for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
	image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
	test_images['test_image_rings'] = np.sin(10 * np.pi * image_grid / N ) * np.cos(10 * np.pi * image_grid / N)

	"Make circular test image"
	image_grid = np.mgrid[:N, :N]
	for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
	image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
	test_images['test_image_circle'] = 1 - gaussian_filter(image_grid, N / 4, 5)

	"Make linear test image"
	test_image = np.zeros((N, N))
	for i in range(4): test_image += np.eye(N, N, k=5-i)
	test_images['test_image_line'] = test_image

	"Make crossed test image"
	test_image = np.zeros((N, N))
	for i in range(4): test_image += np.eye(N, N, k=5-i)
	for i in range(4): test_image += np.rot90(np.eye(N, N, k=5-i))
	test_images['test_image_cross'] = np.where(test_image != 0, 1, 0)

	"Make noisy test image"
	test_images['test_image_noise'] = np.random.random((N, N))

	"Make checkered test image"
	test_images['test_image_checker'] = data.checkerboard()

	return test_images


def test_string_functions():

	from utilities import check_string, check_file_name

	string = "/dir/folder/test_file_SHG.pkl"

	assert check_string(string, -2, '/', 'folder') == "/dir/test_file_SHG.pkl"
	assert check_file_name(string, 'SHG', 'pkl') == "/dir/folder/test_file"


def test_numeric_functions():

	from utilities import unit_vector, numpy_remove, nanmean, ring

	vector = np.array([-3, 2, 6])
	answer = np.array([-0.42857143,  0.28571429,  0.85714286])
	u_vector = unit_vector(vector)

	assert np.sum(u_vector - answer) <= THRESH

	vector_array = np.array([[3, 2, 6], [1, 2, 5], [4, 2, 5], [-7, -1, 2]])

	u_vector_array = unit_vector(vector_array)

	assert np.array(vector_array).shape == u_vector_array.shape

	array_1 = np.arange(50)
	array_2 = array_1 + 20
	answer = np.arange(20)

	edit_array = numpy_remove(array_1, array_2)

	assert abs(answer - edit_array).sum() <= THRESH

	array_nan = np.array([2, 3, 1, np.nan])

	assert nanmean(array_nan) == 2

	ring_answer = np.array([[0, 0, 0, 0, 0],
						 [0, 1, 1, 1, 0],
						 [0, 1, 0, 1, 0],
						 [0, 1, 1, 1, 0],
						 [0, 0, 0, 0, 0]])

	ring_filter = ring(np.zeros((5, 5)), [2, 2], [1], 1)

	assert abs(ring_answer - ring_filter).sum() <= THRESH


def test_FIRE():

	from extraction import (check_2D_arrays, distance_matrix, branch_angles, 
						cos_sin_theta_2D)

	pos_2D = np.array([[1, 3],
					   [4, 2],
					   [1, 5]])

	indices = check_2D_arrays(pos_2D, pos_2D + 1.5, 2)
	assert indices[0] == 2
	assert indices[1] == 0

	answer_d_2D = np.array([[[0, 0], [3, -1], [0, 2]],
							[[-3, 1], [0, 0], [-3, 3]],
							[[0, -2], [3, -3], [0, 0]]])
	answer_r2_2D = np.array([[0, 10, 4], 
							 [10, 0, 18], 
							 [4, 18, 0]])
	d_2D, r2_2D = distance_matrix(pos_2D)

	assert abs(answer_d_2D - d_2D).sum() <= THRESH
	assert abs(answer_r2_2D - r2_2D).sum() <= THRESH

	direction = np.array([1, 0])
	vectors = d_2D[([2, 0], [0, 1])]
	r = np.sqrt(r2_2D[([2, 0], [0, 1])])

	answer_cos_the = np.array([0, 0.9486833])
	cos_the = branch_angles(direction, vectors, r)
	assert abs(answer_cos_the - cos_the).sum() <= THRESH


def test_images():

	from skimage.morphology import local_maxima

	from utilities import ring
	from extraction import new_branches

	images = create_images()

	image = images['test_image_cross'] * 5
	ring_filter = ring(np.zeros(image.shape), [10, 10], np.arange(2, 3), 1)
	branch_coord, branch_vector, branch_r = new_branches(image, np.array([50, 50]), ring_filter)

