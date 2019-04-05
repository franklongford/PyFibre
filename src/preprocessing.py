"""
PyFibre
Preprocessing Library 

Created by: Frank Longford
Created on: 18/02/2019

Last Modified: 18/02/2019
"""

import sys, os, time
import numpy as np
import scipy as sp

from skimage import img_as_float, io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.exposure import rescale_intensity

import utilities as ut


def import_image(image_name):
	"Image importer able to automatically deal with stacks and mixed SHG/PL image types"

	image_orig = ut.load_image(image_name)
	print("Input image shape = {}".format(image_orig.shape))

	if '-pl-shg' in image_name.lower():

		if image_orig.ndim == 4:
			print("Number of image types = {}".format(image_orig.shape[0]))
			shape_check = (image_orig.shape[0] == 3)
			if not shape_check: raise IOError

			image_shape = image_orig[0].shape
			smallest_axis = np.argmin(image_shape)
			xy_dim = tuple(x for i, x in enumerate(image_shape) if i != smallest_axis)

			print("Size of image = {}".format(xy_dim))
			print("Number of stacks = {}".format(image_shape[smallest_axis]))
	
			image_shg = np.mean(image_orig[0], axis=smallest_axis)
			image_pl = np.mean(image_orig[1], axis=smallest_axis)
			image_tran = np.mean(image_orig[2], axis=smallest_axis)

		elif image_orig.ndim == 3:
			image_shape = image_orig.shape
			smallest_axis = np.argmin(image_shape)
			xy_dim = tuple(x for i, x in enumerate(image_shape) if i != smallest_axis)

			print("Number of image types = {}".format(image_orig.shape[smallest_axis]))
			shape_check = (image_orig.shape[smallest_axis] == 3)
			if not shape_check: raise IOError

			image_shg = np.take(image_orig, 0, smallest_axis)
			image_pl = np.take(image_orig, 1, smallest_axis)
			image_tran = np.take(image_orig, 2, smallest_axis)

			print("Size of image = {}".format(xy_dim))

		image_shg = clip_intensities(image_shg, p_intensity=(0, 100))
		image_pl = clip_intensities(image_pl, p_intensity=(0, 100))
		image_tran = clip_intensities(image_tran, p_intensity=(0, 100))

		return image_shg, image_pl, image_tran
		
	elif '-pl' in image_name.lower():

		if image_orig.ndim == 4:
			print("Number of image types = {}".format(image_orig.shape[0]))
			shape_check = (image_orig.shape[0] == 2)
			if not shape_check: raise IOError

			image_shape = image_orig[0].shape
			smallest_axis = np.argmin(image_shape)
			xy_dim = tuple(x for i, x in enumerate(image_shape) if i != smallest_axis)

			print("Size of image = {}".format(xy_dim))
			print("Number of stacks = {}".format(image_shape[smallest_axis]))
	
			image_pl = np.mean(image_orig[0], axis=smallest_axis)
			image_tran = np.mean(image_orig[1], axis=smallest_axis)

		elif image_orig.ndim == 3:
			image_shape = image_orig.shape
			smallest_axis = np.argmin(image_shape)
			xy_dim = tuple(x for i, x in enumerate(image_shape) if i != smallest_axis)

			print("Number of image types = {}".format(image_orig.shape[smallest_axis]))
			shape_check = (image_orig.shape[smallest_axis] == 3)
			if not shape_check: raise IOError

			image_pl = np.take(image_orig, 0, smallest_axis)
			image_tran = np.take(image_orig, 1, smallest_axis)

			print("Size of image = {}".format(image_shape))

		image_pl = clip_intensities(image_pl, p_intensity=(0, 100))
		image_tran = clip_intensities(image_tran, p_intensity=(0, 100))

		return image_pl, image_tran

	elif '-shg' in image_name.lower():

		if image_orig.ndim == 4:
			print("Number of image types = {}".format(image_orig.shape[0]))
			shape_check = (image_orig.shape[0] == 2)
			if not shape_check: raise IOError

			image_shape = image_orig[0].shape
			smallest_axis = np.argmin(image_shape)
			xy_dim = tuple(x for i, x in enumerate(image_shape) if i != smallest_axis)

			print("Size of image = {}".format(xy_dim))
			print("Number of stacks = {}".format(image_shape[smallest_axis]))
	
			image_shg = np.mean(image_orig[1], axis=smallest_axis)

		elif image_orig.ndim == 3:
			image_shape = image_orig.shape
			smallest_axis = np.argmin(image_shape)
			xy_dim = tuple(x for i, x in enumerate(image_shape) if i != smallest_axis)

			print("Size of image = {}".format(xy_dim))
			print("Number of stacks = {}".format(image_shape[smallest_axis]))

			image_shg = np.mean(image_orig, axis=smallest_axis)

		else: 
			print("Size of image = {}".format(image_orig.shape))
			image_shg = image_orig

		image_shg = clip_intensities(image_shg, p_intensity=(0, 100))

		return image_shg

	raise IOError


def load_shg_pl(input_file_names):
	"Load in SHG and PL files from file name tuple"

	image_stack = [None, None, None]

	for filename in input_file_names:
		if '-pl-shg' in filename.lower():
			image_stack[0], image_stack[1], image_stack[2] = import_image(filename)
		elif '-shg' in filename.lower(): 
			image_stack[0] = import_image(filename)
		elif '-pl' in filename.lower(): 
			image_stack[1], image_stack[2] = import_image(filename)

	return image_stack


def clip_intensities(image, p_intensity=(1, 98)):
	"""
	Pre-process image to remove outliers, reduce noise and rescale

	Parameters
	----------

	image:  array_like (float); shape=(n_y, n_x)
		Image to pre-process

	p_intensity: tuple (float); shape=(2,)
		Percentile range for intensity rescaling (used to remove outliers)


	Returns
	-------

	image:  array_like (float); shape=(n_y, n_x)
		Pre-processed image

	"""

	low, high = np.percentile(image, p_intensity)
	image = rescale_intensity(image, in_range=(low, high), out_range=(0.0, 1.0))

	return image


def nl_means(image, p_denoise=(5, 35)):
	"""
	Non-local means denoise algorithm using estimate of Gaussian noise

	Parameters
	----------

	image:  array_like (float); shape=(n_y, n_x)
		Image to pre-process
	
	p_denoise: tuple (float); shape=(2,)
		Parameters for non-linear means denoise algorithm (used to remove noise)

	Returns
	-------

	image:  array_like (float); shape=(n_y, n_x)
		Pre-processed image

	"""

	sigma = estimate_sigma(image)
	image = denoise_nl_means(image, patch_size=p_denoise[0], patch_distance=p_denoise[1],
				fast_mode=True, h = 1.2 * sigma, sigma=sigma, multichannel=False)

	return image


def select_samples(full_set, area, n_sample):
	"""
	Selects n_sample random sections of image stack full_set

	Parameters
	----------

	full_set:  array_like (float); shape(n_frame, n_y, n_x)
		Full set of n_frame images

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	data_set:  array_like (float); shape=(n_sample, 2, n_y, n_x)
		Sampled areas

	indices:  array_like (float); shape=(n_sample, 2)
		Starting points for random selection of full_set

	"""
	
	if full_set.ndim == 2: full_set = full_set.reshape((1,) + full_set.shape)

	n_frame = full_set.shape[0]
	n_y = full_set.shape[1]
	n_x = full_set.shape[2]

	data_set = np.zeros((n_sample, n_frame, area, area))

	pad = area // 2

	indices = np.zeros((n_sample, 2), dtype=int)

	for n in range(n_sample):

		try: start_x = np.random.randint(pad, n_x - pad)
		except: start_x = pad
		try: start_y = np.random.randint(pad, n_y - pad) 
		except: start_y = pad

		indices[n][0] = start_x
		indices[n][1] = start_y

		data_set[n] = full_set[:, start_y-pad: start_y+pad, 
					  start_x-pad: start_x+pad]

	return data_set.reshape(n_sample * n_frame, area, area), indices
