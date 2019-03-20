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

	print(image_orig.shape)

	if image_orig.ndim > 2:

		if image_orig.ndim == 4:

			if image_orig.shape[0] == 2:

				image = np.mean(image_orig[0], axis=-1)
				image_tran = np.mean(image_orig[1], axis=-1)
			
				image = image / image.max()
				image_tran = image_tran / image_tran.max()

				return image, image_tran

			elif image_orig.shape[0] == 3:
	
				image_shg = np.mean(image_orig[0],  axis=-1)
				image_pl = np.mean(image_orig[1],  axis=-1)
				image_tran = np.mean(image_orig[2],  axis=-1)
			
				image_shg = image_shg / image_shg.max()
				image_pl = image_pl / image_pl.max()
				image_tran = image_tran / image_tran.max()

				return image_shg, image_pl, image_tran

		else:
			smallest_axis = np.argmin(image_orig.shape)

			image_euclid = np.sqrt(np.sum(image_orig**2, axis=smallest_axis))
			image_mean = np.mean(image_orig, axis=smallest_axis)
			image_mix = np.sqrt(image_mean * image_euclid)

			image = image_mean / image_mix.max()

			return image

	else: return image_orig / image_orig.max()

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
			image_stack[1] = import_image(filename)[0]
			image_stack[2] = import_image(filename)[1]

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
	image = rescale_intensity(image, in_range=(low, high))
	image /= image.max()

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
