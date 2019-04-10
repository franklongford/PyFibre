"""
PyFibre
Image Tools Library 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 18/02/2019
"""

import sys, os, time
import numpy as np
import scipy as sp

from scipy.ndimage.filters import gaussian_filter

from skimage.feature import greycoprops
from skimage.color import grey2rgb, rgb2grey, rgb2hsv, hsv2rgb

import utilities as ut
from extraction import branch_angles


def fourier_transform_analysis(image, n_split=1, sigma=None, nbins=200):
	"""
	Calculates fourier amplitude spectrum for image

	Parameters
	----------

	image:  array_like (float); shape=(n_x, n_y)
	Image to analyse

	Returns
	-------

	angles:  array_like (float); shape=(n_bins)
	Angles corresponding to fourier amplitudes

	fourier_spec:  array_like (float); shape=(n_bins)
	Average Fouier amplitudes of FT of image_shg

	"""

	if sigma != None: image = filters.gaussian_filter(image, sigma)

	sampled_regions = matrix_split(image, n_split, n_split)

	for region in sampled_regions:

		image_fft = np.fft.fft2(region)
		image_fft[0][0] = 0
		image_fft = np.fft.fftshift(image_fft)

		real = np.real(image_fft)
		imag = np.imag(image_fft)

		magnitude = np.abs(image_fft)
		phase = np.angle(image_fft, deg=True)



		image_grid = np.mgrid[:region.shape[0], :region.shape[1]]
		for i in range(2): 
		    image_grid[i] -= region.shape[0] * np.array(2 * image_grid[i] / region.shape[0],
			                dtype=int)
		image_radius = np.sqrt(np.sum(image_grid**2, axis=0))

		angles = image_grid[0] / image_radius
		angles = (np.arccos(angles) * 360 / np.pi)
		angles[0][0] = 0
		angles = np.fft.fftshift(angles)

		print(angles.max(), angles.min())


		sdi = np.mean(fourier_spec) / np.max(fourier_spec)

	return angles, fourier_spec, sdi


def tensor_analysis(tensor):
	"""
	tensor_analysis(tensor)

	Calculates eigenvalues and eigenvectors of average tensor over area^2 pixels for n_samples

	Parameters
	----------

	tensor:  array_like (float); shape(nframe, nx, ny, 2, 2)
		Average tensor over area under examination 

	Returns
	-------

	tot_anis:  array_like (float); shape=(n_frame, nx, ny)
		Difference between eigenvalues of average tensors

	tot_angle:  array_like (float); shape=(n_frame, nx, ny)
		Angle of dominant eigenvector of average tensors

	tot_energy:  array_like (float); shape=(n_frame, nx, ny)
		Determinent of eigenvalues of average tensors

	"""

	if tensor.ndim == 2: tensor = tensor.reshape((1,) + tensor.shape)

	eig_val, eig_vec = np.linalg.eigh(tensor)

	eig_diff = np.diff(eig_val, axis=-1).max(axis=-1)
	eig_sum = eig_val.sum(axis=-1)
	indicies = np.nonzero(eig_sum)

	tot_anis = np.zeros(tensor.shape[:-2])
	tot_anis[indicies] += eig_diff[indicies] / eig_sum[indicies]

	tot_angle = 0.5 * np.arctan2(2 * tensor[..., 1, 0], (tensor[..., 1, 1] - tensor[..., 0, 0])) / np.pi * 180
	#tot_angle = np.arctan2(tensor[..., 1, 0], tensor[..., 1, 1]) / np.pi * 180
	tot_energy = np.trace(np.abs(tensor), axis1=-2, axis2=-1)

	return tot_anis, tot_angle, tot_energy


def angle_analysis(angles, weights, N=200):

	angle_hist, _ = np.histogram(angles.flatten(), bins=N,
					weights=weights.flatten(), density=True)
	angle_sdi = angle_hist.mean() / angle_hist.max()

	return angle_sdi


def fibre_analysis(tot_fibres, verbose=False):

	fibre_lengths = np.empty((0,), dtype='float64')
	fibre_waviness = np.empty((0,), dtype='float64')
	fibre_angles = np.empty((0,), dtype='float64')

	for fibre in tot_fibres:

		start = fibre.node_list[0]
		end = fibre.node_list[-1]

		if verbose: print("N nodes", len(fibre.node_list), "Length", fibre.fibre_l, 
					"Displacement", fibre.euclid_l, "Direction", fibre.direction, "\n")

		fibre_lengths = np.concatenate((fibre_lengths, [fibre.fibre_l]))
		fibre_waviness = np.concatenate((fibre_waviness, [fibre.euclid_l / fibre.fibre_l]))

		cos_the = branch_angles(fibre.direction, np.array([[0, 1]]), np.ones(1))
		fibre_angles = np.concatenate((fibre_angles, np.arccos(cos_the) * 180 / np.pi))

	return fibre_lengths, fibre_waviness, fibre_angles


def greycoprops_edit(P, prop='contrast'):


	(num_level, num_level2, num_dist, num_angle) = P.shape

	assert num_level == num_level2
	assert num_dist > 0
	assert num_angle > 0

	# create weights for specified property
	I, J = np.ogrid[0:num_level, 0:num_level]
	if prop == 'IDM': weights = 1. / (1. + abs(I - J))
	elif prop in ['variance', 'cluster', 'entropy']: pass
	else: return greycoprops(P, prop)

	# normalize each GLCM
	P = P.astype(np.float64)
	glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
	glcm_sums[glcm_sums == 0] = 1
	P /= glcm_sums

	if prop in ['IDM']:
		weights = weights.reshape((num_level, num_level, 1, 1))
		results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]

	elif prop == 'variance':
		I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
		J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
		diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
		diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

		results = np.apply_over_axes(np.sum, (P * (diff_i * diff_j)),
		                         axes=(0, 1))[0, 0]

	elif prop == 'cluster':
		I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
		J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
		diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
		diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

		results = np.apply_over_axes(np.sum, (P * (I + J - diff_i - diff_j)),
		                         axes=(0, 1))[0, 0]

	elif prop == 'entropy':
		nat_log = np.log(P)

		mask_0 = P < 1e-15
		mask_0[P < 1e-15] = True
		nat_log[mask_0] = 0

		results = np.apply_over_axes(np.sum, (P * (- nat_log)),
		                         axes=(0, 1))[0, 0]


	return results

