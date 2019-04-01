"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import sys, os
import numpy as np
import scipy as sp
import networkx as nx

from scipy import interpolate
from scipy.ndimage import sobel
from scipy.ndimage.filters import gaussian_filter

from skimage.feature import structure_tensor, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import apply_hysteresis_threshold, sobel
from skimage.filters import threshold_otsu, threshold_li, threshold_isodata, threshold_mean


def gaussian(image, sigma=None):

	if sigma != None: return gaussian_filter(image, sigma=sigma)
	else: return image


def spiral_tv(image):

	pass


def tubeness(image, sigma=None):

	H_elems = hessian_matrix(image, order="xy", sigma=sigma, mode='wrap')
	H_eigen = hessian_matrix_eigvals(H_elems)
	tube = np.where(H_eigen[1] < 0, abs(H_eigen[1]), 0)

	return tube


def curvelet(image):

	pass


def vesselness(eig1, eig2, beta1=0.1, beta2=0.1):

	A = np.exp(-(eig1/eig2)**2 / (2 * beta1))
	B = (1 - np.exp(- (eig1**2 + eig2**2) / (2 * beta2)))

	return A * B


def hysteresis(image, alpha=1.0):
	""

	low = np.min([alpha * threshold_mean(image), threshold_li(image)])
	high = threshold_isodata(image)

	threshold = apply_hysteresis_threshold(image, low, high)

	return threshold


def derivatives(image, rank=1):
	"""
	Returns derivates of order "rank" for imput image at each pixel

	Parameters
	----------

	image:  array_like (float); shape(n_y, n_x)
		Image to analyse

	rank:  int (optional)
		Order of derivatives to return (1 = first order, 2 = second order)

	Returns
	-------

	derivative:  array_like (float); shape=(2 or 4, n_y, n_x)
		First or second order derivatives at each image pixel
	"""

	derivative = np.zeros(((2,) + image.shape))
	derivative[0] += np.nan_to_num(np.gradient(image, edge_order=1, axis=-2))
	derivative[1] += np.nan_to_num(np.gradient(image, edge_order=1, axis=-1))

	if rank == 2:
		second_derivative = np.zeros(((4,) + image.shape))
		second_derivative[0] += np.nan_to_num(np.gradient(derivative[0], edge_order=1, axis=-2))
		second_derivative[1] += np.nan_to_num(np.gradient(derivative[1], edge_order=1, axis=-2))
		second_derivative[2] += np.nan_to_num(np.gradient(derivative[0], edge_order=1, axis=-1))
		second_derivative[3] += np.nan_to_num(np.gradient(derivative[1], edge_order=1, axis=-1))

		return second_derivative

	else: return derivative


def form_nematic_tensor(image, sigma=None, size=None):
	"""
	form_nematic_tensor(dx_shg, dy_shg)

	Create local nematic tensor n for each pixel in dx_shg, dy_shg

	Parameters
	----------

	dx_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to x axis for each pixel

	dy_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to y axis for each pixel

	Returns
	-------

	n_vector:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)	

	"""

	if image.ndim == 2: image = image.reshape((1,) + image.shape)
	nframe = image.shape[0]

	dx_shg, dy_shg = derivatives(image)
	r_xy_2 = (dx_shg**2 + dy_shg**2)
	indicies = np.where(r_xy_2 > 0)

	nxx = np.zeros(dx_shg.shape)
	nyy = np.zeros(dx_shg.shape)
	nxy = np.zeros(dx_shg.shape)

	nxx[indicies] += dy_shg[indicies]**2 / r_xy_2[indicies]
	nyy[indicies] += dx_shg[indicies]**2 / r_xy_2[indicies]
	nxy[indicies] -= dx_shg[indicies] * dy_shg[indicies] / r_xy_2[indicies]

	if sigma != None:
		for frame in range(nframe):
			nxx[frame] = gaussian_filter(nxx[frame], sigma=sigma)
			nyy[frame] = gaussian_filter(nyy[frame], sigma=sigma)
			nxy[frame] = gaussian_filter(nxy[frame], sigma=sigma)

	n_tensor = np.stack((nxx, nxy, nxy, nyy), -1).reshape(nxx.shape + (2,2))
	if nframe == 1: n_tensor = n_tensor.reshape(n_tensor.shape[1:])

	return n_tensor


def form_structure_tensor(image, sigma=0.0001, size=None):
	"""
	form_structure_tensor(image)

	Create local structure tensor n for each pixel in image

	Parameters
	----------

	dx_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to x axis for each pixel

	dy_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to y axis for each pixel

	Returns
	-------

	j_tensor:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
		2x2 structure tensor for each pixel in image stack	

	"""

	if image.ndim == 2: image = image.reshape((1,) + image.shape)
	nframe = image.shape[0]

	jxx = np.zeros(image.shape)
	jxy = np.zeros(image.shape)
	jyy = np.zeros(image.shape)

	for frame in range(nframe):
		jxx[frame], jxy[frame], jyy[frame] = structure_tensor(image[frame], sigma=sigma)

	j_tensor = np.stack((jxx, jxy, jxy, jyy), -1).reshape(jxx.shape + (2,2))
	if nframe == 1: j_tensor = j_tensor.reshape(j_tensor.shape[1:])

	return j_tensor


def form_hessian_tensor(image, sigma=None, size=None):
	"""
	form_hessian_tensor(image)

	Create local hessian tensor n for each pixel in image

	Parameters
	----------

	dx_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to x axis for each pixel

	dy_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to y axis for each pixel

	Returns
	-------

	H_tensor:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
		2x2 hessian tensor for each pixel in image stack	

	"""

	if image.ndim == 2: image = image.reshape((1,) + image.shape)
	nframe = image.shape[0]

	dxdx = np.zeros(image.shape)
	dxdy = np.zeros(image.shape)
	dydy = np.zeros(image.shape)

	#dxdx, dxdy, dydx, dyy = derivatives(image, rank=2)
	for frame in range(nframe):
		dxdx[frame], dxdy[frame], dydy[frame] = hessian_matrix(image[frame], order="xy", sigma=sigma)

	H_tensor = np.stack((dxdx, dxdy, dxdy, dydy), -1).reshape(dxdx.shape + (2,2))
	if nframe == 1: H_tensor = H_tensor.reshape(H_tensor.shape[1:])

	return H_tensor


def get_curvature(j_tensor, H_tensor):
	"""
	Return Gaussian and Mean curvature at each pixel from structure and hessian tensors

	Parameters
	----------

	j_tensor:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
		2x2 structure tensor for each pixel in image stack

	H_tensor:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
		2x2 hessian tensor for each pixel in image stack

	Returns
	-------

	gauss_curvature: array_like (float); shape(nframe, n_y, n_x)
		Gaussian curvature at each image pixel

	mean_curvature: array_like (float); shape(nframe, n_y, n_x)
		Mean curvature at each image pixel
	"""

	ad_H_tensor = ut.adjoint_mat(H_tensor)

	denominator = (1 + j_tensor[...,0,0] + j_tensor[...,1,1])
	gauss_curvature = np.linalg.det(H_tensor) / denominator**2

	numerator = - 2 * j_tensor[...,0,1] * H_tensor[...,0,1]
	numerator += (1 + j_tensor[...,1,1]) * H_tensor[...,0,0]
	numerator += (1 + j_tensor[...,0,0]) * H_tensor[...,1,1]

	mean_curvature =  numerator / (2 * denominator**1.5)

	return np.nan_to_num(gauss_curvature), np.nan_to_num(mean_curvature)
