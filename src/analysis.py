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

from skimage.color import grey2rgb, rgb2grey, rgb2hsv, hsv2rgb

import utilities as ut


def set_HSB(image, hue, saturation=1, brightness=1):
	""" Add color of the given hue to an greyscale image.

	By default, set the saturation to 1 so that the colors pop!
	"""
	rgb = grey2rgb(image)
	hsv = rgb2hsv(rgb)

	hsv[..., 0] = hue
	hsv[..., 1] = saturation
	hsv[..., 2] = brightness

	return hsv2rgb(hsv)


def fourier_transform_analysis(image, sigma=None, n_sample=100, size=100, nbins=200):
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

    if sigma != None: image = gaussian_filter(image, sigma)
    
    image_fft = np.fft.fft2(image)
    image_ifft =  np.fft.ifft2(image_fft)
    
    fft_angle = np.angle(image_fft, deg=True)
    fft_magnitude = np.where(fft_angle == 0, 0, np.abs(image_fft))
    fft_order = np.argsort(fft_angle.flatten())
    
    sdi = np.mean(fft_magnitude) / np.max(fft_magnitude)

    return fft_angle.flatten()[fft_order], fft_magnitude.flatten()[fft_order], sdi


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


def angle_analysis(angles): 

	pass


