"""
ColECM: Collagen ExtraCellular Matrix Simulation
UTILITIES ROUTINE 

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 12/04/2018
"""

import numpy as np

import sys, os, pickle

from skimage import img_as_float, io

SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)
SQRTPI = np.sqrt(np.pi)


class NoiseError(Exception):
    
    def __init__(self, noise, thresh):

    	self.noise = noise
    	self.thresh = thresh
    	self.message = "Image too noisy ({} > {})".format(noise, thresh)


def logo():

	logo_text = ' ' + '_' * 54 + '\n'
	logo_text += "|_______|" + ' ' * 14 + "|_______|" + ' ' * 14 + "|_______|" + '\n'
	logo_text += " \\_____/" + ' ' * 16 + "\\_____/"  + ' ' * 16 + "\\_____/"  + '\n'
	logo_text += "  | | |    ___         ___                      | | |" + '\n'
	logo_text += "  | | |    |  \       |    .  |                 | | |" + '\n'
	logo_text += "  | | |    |__/       |__     |__    __  __     | | |" + '\n'
	logo_text += "  | | |    |    |  |  |    |  |  \  |   |__|    | | |" + '\n'
	logo_text += "  | | |    |    \__|  |    |  |__/  |   |__     | | |" + '\n'
	logo_text += "  | | |          __/                            |_|_|" + '\n'
	logo_text += "  |_|_|                                         |_|_|" + '\n'
	logo_text += " /_____\\" + ' ' * 16 + "/_____\\"  + ' ' * 16 + "/_____\\"  + '\n'
	logo_text += "|_______|" + '_' * 14 + "|_______|" + '_' * 14 + "|_______|" + '  v1.2' + '\n'
	logo_text += "\n              Fibrous Tissue Image Toolkit\n"

	return logo_text


def check_string(string, pos, sep, word):

	if sep in string: 
		temp_string = string.split(sep)
		if temp_string[pos] == word: temp_string.pop(pos)
		string = sep.join(temp_string)

	return string


def check_file_name(file_name, file_type="", extension=""):
	"""
	check_file_name(file_name, file_type="", extension="")
	
	Checks file_name for file_type or extension

	"""

	file_name = check_string(file_name, -1, '.', extension)
	file_name = check_string(file_name, -1, '_', file_type)
	
	return file_name


def load_image(image_name):

	image = io.imread(image_name).astype(float)

	return image


def save_npy(file_path, array):
	"""
	save_npy(file_path, array)

	General purpose algorithm to save an array to a npy file

	Parameters
	----------

	file_path:  str
		Path name of npy file
	array:  array_like (float);
		Data array to be saved
	"""

	np.save(file_path, array)


def load_npy(file_path, frames=[]):
	"""
	load_npy(file_path, frames=[])

	General purpose algorithm to load an array from a npy file

	Parameters
	----------

	file_path:  str
		Path name of npy file
	frames:  int, list (optional)
		Trajectory frames to load

	Returns
	-------
	array:  array_like (float);
		Data array to be loaded
	"""

	if len(frames) == 0: array = np.load(file_path + '.npy')
	else: array = np.load(file_path + '.npy')[frames]

	return array


def save_region(regions, file_name):
	"Saves scikit image regions as pickled file"

	with open('{}.pkl'.format(file_name), 'wb') as outfile:
		pickle.dump(regions, outfile, pickle.HIGHEST_PROTOCOL)


def load_region(file_name):
	"Loads pickled scikit image regions"

	with open('{}.pkl'.format(file_name), 'rb') as infile:
		regions = pickle.load(infile)

	return regions


def numpy_remove(list1, list2):
	"""
	numpy_remove(list1, list2)

	Deletes overlapping elements of list2 from list1
	"""

	return np.delete(list1, np.where(np.isin(list1, list2)))


def unit_vector(vector, axis=-1):
	"""
	unit_vector(vector, axis=-1)

	Returns unit vector of vector
	"""

	vector = np.array(vector)
	magnitude_2 = np.resize(np.sum(vector**2, axis=axis), vector.shape)
	u_vector = np.sqrt(vector**2 / magnitude_2) * np.sign(vector)

	return u_vector


def rand_vector(n): 
	"""
	rand_vector(n)
	
	Returns n dimensional unit vector, components of which lie in the range -1..1

	"""

	return unit_vector(np.random.random((n)) * 2 - 1) 


def remove_element(a, array): 
	"""
	remove_element(a, array)
	
	Returns new array without element a

	"""

	return np.array([x for x in array if x != a])



def gaussian(x, mean, std):
	"""
	Return value at position x from Gaussian distribution with centre mean and standard deviation std
	"""

	return np.exp(-(x-mean)**2 / (2 * std**2)) / (SQRT2 * std * SQRTPI)


def dx_gaussian(x, mean, std):
	"""
	Return derivative of value at position x from Gaussian distribution with centre mean and standard deviation std
	"""

	return (mean - x) / std**2 * gaussian(x, mean, std)


def reorder_array(array):
	"""
	reorder_array(array)

	Inverts 3D array so that outer loop is along z axis
	"""

	return np.moveaxis(array, (2, 0, 1), (0, 1, 2))


def move_array_centre(array, centre):
	"""
	move_array_centre(array, centre)

	Move top left corner of ND array to centre index
	"""

	n_dim = np.array(centre).shape[0]

	for i, ax in enumerate(range(n_dim)): array = np.roll(array, centre[i], axis=ax)

	return array


def nanmean(array_like, weights=None):

	try: 
		assert weights == None
		weights = np.ones(array_like.shape)

	except (AssertionError, ValueError): pass

	indices = ~np.isnan(array_like)

	try: average = np.average(array_like[indices], weights=weights[indices])
	except ZeroDivisionError: average = None

	return average


def adjoint_mat(matrix):

	xx = matrix[...,0,0]
	xy = matrix[...,0,1]
	yx = matrix[...,1,0]
	yy = matrix[...,1,1]

	mat_ad = np.stack((-yy, yx, xy, -xx), -1).reshape(xx.shape + (2,2))

	return mat_ad


def ring(image, index, sizes, value):
    
    index = np.array(index)
    sizes = np.array(sizes)
    
    for size in sizes:
        indices = np.concatenate((index - size, index + size))

        if indices[0] >= 0: 
            image[indices[0], 
                  max([indices[1], 0]): min([indices[3], image.shape[1]])+1] = value
        if indices[2] < image.shape[0]: 
            image[indices[2], 
                  max([indices[1], 0]): min([indices[3], image.shape[1]])+1] = value
        if indices[1] >= 0: 
            image[max([indices[0], 0]): min([indices[2], image.shape[0]])+1,
                 indices[1]] = value
        if indices[3] < image.shape[1]: 
            image[max([indices[0], 0]): min([indices[2], image.shape[0]])+1,
                 indices[3]] = value

    return image


def conv_coord(array):
    
    return (array[:,0], array[:,1])


def clear_border(image, thickness=1):

	for i in range(thickness):
		image[:, 0 + i] = 0
		image[0 + i, :] = 0
		image[:, -(1 + i)] = 0
		image[-(1 + i), :] = 0

	return image


def flatten_list(list_of_lists):

	flat_list = [val for sublist in list_of_lists for val in sublist]

	return flat_list


def extract_prefix(image_name, label):
	"Extract the prefix of image_name, before label"
	directory = '/'.join(image_name.split('/')[ : -1])
	filename = image_name.split('/')[-1]
	filename_copy = filename.lower()

	index = filename_copy.index(label.lower())
	prefix = directory + '/' + filename[ : index]

	return prefix


def get_files_prefixes(file_list, label):
	"Get the file path and file prefix of all files containin label"

	files = [(filename,) for filename in file_list \
	                 if label in filename.split('/')[-1].lower()]
	prefixes = [extract_prefix(filename[0], label) for filename in files]

	removed_files = []
	prefix_count = []

	return files, prefixes

    
def get_image_lists(input_files):
	"Automatically find all combined PL-SHG files or match up individual images if seperate"

	shg_pl_files, shg_pl_prefixes = get_files_prefixes(input_files, '-pl-shg')

	for filename in shg_pl_files: input_files.remove(filename[0])

	shg_files, shg_prefixes = get_files_prefixes(input_files, '-shg')
	pl_files, pl_prefixes = get_files_prefixes(input_files, '-pl')

	for i, prefix in enumerate(shg_prefixes):

		if prefix not in shg_pl_prefixes:
			indices = [j for j, pl_prefix in enumerate(pl_prefixes) if prefix in pl_prefix]

			if len(indices) > 0: shg_pl_files += [(shg_files[i][0], pl_files[indices[0]][0])]
			else: shg_pl_files += [(shg_files[i][0],)]

			shg_pl_prefixes += [prefix]


	return shg_pl_files, shg_pl_prefixes


####### OBSOLETE ########

def gaussian_filter(histogram, std, r, n_xyz):

	"Get indicies and intensity of non-zero histogram grid points"
	indices = np.argwhere(histogram)
	intensity = histogram[np.where(histogram)]

	"Generate blank image"
	image = np.zeros(n_xyz[:2])

	for i, index in enumerate(indices):
		"Performs the full mapping"
		if n_dim == 2: r_shift = move_array_centre(r, index[::-1])
		elif n_dim == 3: r_shift = move_array_centre(r[index[0]], index[1:])
		image += np.reshape(gaussian(r_shift.flatten(), 0, std), n_xyz[:2]) * intensity[i]

	image = image.T

	return image


def bubble_sort(array, key):
	"""
	bubble_sort(array, key)

	Sorts array and key by order of elements of key
	"""

	for passnum in range(len(array)-1, 0, -1):
		for i in range(passnum):
			if key[i] > key[i+1]:
				temp = array[i]
				array[i] = array[i+1]
				array[i+1] = temp

				temp = key[i]
				key[i] = key[i+1]
				key[i+1] = temp


def dx_dy_shg(histogram, std, n_xyz, dxdydz, r, non_zero):
	"""
	dx_dy_shg(histogram, std, n_xyz, dxdydz, r, non_zero)

	Create Gaussian convoluted image from a set of bead positions

	Parameter
	---------

	histogram:  array_like (int); shape=(n_x, n_y)
		Discretised distribution of pos_x and pos_y

	std:  float
		Standard deviation of Gaussian distribution

	n_xyz:  tuple (int); shape(n_dim)
		Number of pixels in each image dimension

	dxdydz:  array_like (float); shape=(n_x, n_y, n_z)
		Matrix of distances along x y and z axis in pixels with cutoff radius applied

	r_cut:  array_like (float); shape=(n_x, n_y)
		Matrix of radial distances between pixels with cutoff radius applied

	non_zero:  array_like (float); shape=(n_x, n_y)
		Filter representing indicies to use in convolution

	Returns
	-------

	dx_grid:  array_like (float); shape=(n_y, n_x)
		Matrix of derivative of image intensity with respect to x axis for each pixel

	dy_grid:  array_like (float); shape=(n_y, n_x)
		Matrix of derivative of image intensity with respect to y axis for each pixel

	"""

	"Get indicies and intensity of non-zero histogram grid points"
	indices = np.argwhere(histogram)
	intensity = histogram[np.where(histogram)]
	n_dim = len(n_xyz)

	if n_dim == 3: 
		r = ut.reorder_array(r)
		non_zero = ut.reorder_array(non_zero)
		dxdydz = np.moveaxis(dxdydz, (0, 3, 1, 2), (0, 1, 2, 3))

	n_dim = len(n_xyz)
	"Generate blank image"
	dx_grid = np.zeros(n_xyz[:2])
	dy_grid = np.zeros(n_xyz[:2])

	for i, index in enumerate(indices):
	
		if n_dim == 2:
			r_shift = ut.move_array_centre(r, index)
			non_zero_shift = ut.move_array_centre(non_zero, index)
			dx_shift = ut.move_array_centre(dxdydz[0], index)
			dy_shift = ut.move_array_centre(dxdydz[1], index)

		elif n_dim == 3:

			r_shift = ut.move_array_centre(r[-index[0]], index[1:])
			non_zero_shift = ut.move_array_centre(non_zero[-index[0]], index[1:])
			dx_shift = ut.move_array_centre(dxdydz[0][-index[0]], index[1:])
			dy_shift = ut.move_array_centre(dxdydz[1][-index[0]], index[1:])
			
		dx_grid[np.where(non_zero_shift)] += (ut.dx_gaussian(r_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dx_shift[np.where(non_zero_shift)].flatten() / r_shift[np.where(non_zero_shift)].flatten())
		dy_grid[np.where(non_zero_shift)] += (ut.dx_gaussian(r_shift[np.where(non_zero_shift)].flatten(), 0, std) * 
							intensity[i] * dy_shift[np.where(non_zero_shift)].flatten() / r_shift[np.where(non_zero_shift)].flatten())

	dx_grid = dx_grid.T
	dy_grid = dy_grid.T

	return dx_grid, dy_grid
