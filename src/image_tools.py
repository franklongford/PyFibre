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

from PIL import Image

from scipy.misc import derivative
from scipy.ndimage import filters, imread
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from scipy.optimize import curve_fit
from scipy.stats import invgauss

from skimage import data, measure, img_as_float, exposure, feature
from skimage.transform import rescale
from skimage.morphology import (disk, dilation)
from skimage.filters import rank, threshold_li, threshold_mean, hessian, threshold_otsu, median
from skimage.color import rgb2hsv, hsv2rgb
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman, estimate_sigma
from skimage.feature import ORB

from sklearn.decomposition import NMF
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import spectral_clustering

import utilities as ut
from filters import tubeness
from extraction import FIRE, adj_analysis


def load_image(image_name):

	image_orig = np.asarray(Image.open(image_name)).astype(np.float32)

	if image_orig.ndim > 2: 
		image = np.sum(image_orig / image_orig.max(axis=-1), axis=0)
	else: image = image_orig# / image_orig.max()

	return image


def set_HSB(image, hue, saturation=1, brightness=1):
	""" Add color of the given hue to an RGB image.

	By default, set the saturation to 1 so that the colors pop!
	"""
	hsv = rgb2hsv(image)

	hsv[..., 0] = hue
	hsv[..., 1] = saturation
	hsv[..., 2] = brightness

	return hsv2rgb(hsv)

def optimise_histogram(image, func, precision = 1E-10, max_it=100):
    
	bins = [25]
	error = []
	optimising = True
	iteration = 1

	while optimising:

		try: 
			hist, X = np.histogram(image[np.nonzero(image)].flatten(), density=True, bins=bins[-1])
			X = X[1:]
			popt_iga, pcov  = curve_fit(func, X, hist)
			error.append(((hist - func(X, *popt_iga))**2).sum() / bins[-1]**2)

			if len(error) >= 2: optimising = (error[-1] >= precision) * (iteration < max_it)
			if optimising: bins.append(bins[-1] + 1)
		    
		except RuntimeError: bins[-1] += 1
		    
		iteration += 1
	    
	hist, X = np.histogram(image[np.nonzero(image)].flatten(), density=True, bins=bins[np.argmin(error)])
	X = X[1:]
	popt_iga, pcov  = curve_fit(func, X, hist)

	return X, hist, popt_iga


def func_invgauss(x, mu, loc, scale): return invgauss.pdf(x, mu, loc, scale)


def preprocess_image(image, clip_limit=None, interval=0.95, threshold=False):

	"Median averaging to remove shot noise"
	image = image / image.max()
	image = median(image, disk(3))

	"Gaussian blur to remove gaussian noise"
	noise = estimate_sigma(image)
	image = filters.gaussian_filter(image, sigma=noise)

	if threshold:

		X, hist, popt_iga = optimise_histogram(image, func_invgauss)
		mu, loc, scale = popt_iga
		mean, var, skew, kurt = invgauss.stats(mu, loc=loc, scale=scale, moments='mvsk')

		clip_high = invgauss.interval(interval, mu=mu, loc=loc, scale=scale)[1]
		clip_low = threshold_mean(image)

		image = np.where(image <= clip_high, image, clip_high)
		image = np.where(image >= clip_low, image, 0)

	if clip_limit != None: image = exposure.equalize_adapthist(image / image.max(), clip_limit=clip_limit)

	return image


def select_samples(full_set, area, n_sample):
	"""
	select_samples(full_set, area, n_sample)

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


def derivatives(image, rank=1, mode='cd'):
	"""
	Returns derivates of order "rank" for imput image at each pixel
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
			nxx[frame] = filters.gaussian_filter(nxx[frame], sigma=sigma)
			nyy[frame] = filters.gaussian_filter(nyy[frame], sigma=sigma)
			nxy[frame] = filters.gaussian_filter(nxy[frame], sigma=sigma)

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
		jxx[frame], jxy[frame], jyy[frame] = feature.structure_tensor(image[frame], sigma=sigma)

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
		dxdx[frame], dxdy[frame], dydy[frame] = feature.hessian_matrix(image[frame], order="xy", sigma=sigma)

	H_tensor = np.stack((dxdx, dxdy, dxdy, dydy), -1).reshape(dxdx.shape + (2,2))
	if nframe == 1: H_tensor = H_tensor.reshape(H_tensor.shape[1:])

	return H_tensor


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


def network_extraction(graph_name, image, ow_graph=False, threads=8):
	"""
	Extract fibre network using modified FIRE algorithm
	"""

	try: Aij = nx.read_gpickle(graph_name + ".pkl")
	except IOError: ow_graph = True

	if ow_graph:
		print("Extracting fibre network using modified FIRE algorithm")	
		image_TB = tubeness(image, 1)
		Aij = FIRE(image_TB, sigma=1, max_threads=threads)
		nx.write_gpickle(Aij, graph_name + ".pkl")

	label_image = np.zeros(image.shape)
	networks = []

	for i, component in enumerate(nx.connected_components(Aij)):
		subgraph = Aij.subgraph(component)
		if subgraph.number_of_nodes() > 3:
			networks.append(subgraph)
			nodes_coord = np.stack((subgraph.nodes[i]['xy'] for i in subgraph.nodes()))
			label_image[nodes_coord[:,0],nodes_coord[:,1]] = (i + 1) 

	n_clusters = len(networks)
	label_image = np.array(label_image, dtype=int)
	areas = np.empty((0,), dtype=float)
	regions = []

	for region in measure.regionprops(label_image): 
	    areas = np.concatenate((areas, [region.filled_area]))
	    regions.append(region.bbox)

	sort_areas = np.argsort(areas)[-n_clusters:]
	sort_regions = []

	for index in sort_areas: sort_regions.append(regions[index])

	return label_image, sort_areas, sort_regions, networks


def network_area(label, iterations=10):

	dilated_image = binary_dilation(label, iterations=iterations)
	filled_image = dilated_image#binary_fill_holes(dilated_image)
	net_area = np.sum(filled_image)

	return net_area, filled_image


def network_analysis(label_image, sorted_areas, networks, n_tensor, anis_map):
	"""
	Analyse extracted fibre network
	"""

	main_network = np.zeros(label_image.shape, dtype=int)
	net_area = np.zeros(len(sorted_areas))
	net_anis = np.zeros(len(sorted_areas))
	net_linear = np.zeros(len(sorted_areas))
	net_degree = np.zeros(len(sorted_areas))
	fibre_waviness = np.zeros(len(sorted_areas))
	network_waviness = np.zeros(len(sorted_areas))
	net_cluster = np.zeros(len(sorted_areas))
	pix_anis = np.zeros(len(sorted_areas))
	solidity = np.zeros(len(sorted_areas))

	for i, n in enumerate(sorted_areas):
		network_matrix = np.zeros(label_image.shape, dtype=int)

		region = measure.regionprops(label_image)[n]
		solidity[i] = region.solidity
		minr, minc, maxr, maxc = region.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		region_anis = anis_map[(indices[0], indices[1])]
		region_tensor = n_tensor[(indices[0], indices[1])]
		network_matrix[(indices[0], indices[1])] += region.image * (i+1)

		net_area[i], network = network_area(region.image)
		indices = np.where(network)
		region_tensor = region_tensor[(indices[0], indices[1])]
		region_anis = region_anis[(indices[0], indices[1])]

		net_anis[i], _ , _ = tensor_analysis(np.mean(region_tensor, axis=(0)))
		pix_anis[i] = np.mean(region_anis)

		if network.shape[0] > 1 and network.shape[1] > 1:
			region = measure.regionprops(np.array(network, dtype=int))[0]
			net_linear[i] += 1 - region.equivalent_diameter / region.perimeter
		main_network += network_matrix

		fibre_waviness[i], network_waviness[i] = adj_analysis(networks[i])
				
		try: net_cluster[i] = nx.average_clustering(networks[i])
		except: net_cluster[i] = None

		try: net_degree[i] = nx.degree_pearson_correlation_coefficient(networks[i])**2
		except: net_degree[i] = None

	coverage = np.count_nonzero(main_network) / main_network.size

	#sys.exit()

	return (net_area, net_anis, net_linear, net_cluster, net_degree, fibre_waviness, 
			network_waviness, pix_anis, coverage, solidity)


def get_snr_estimates(image, guess, inc = 1E-3):

	clip_limit = [guess[0] - inc, guess[0] + inc]
	sigma = [guess[1] - inc, guess[1] + inc]

	snr = np.zeros((2, 2))
	jacobian = np.zeros(2)
	#hessian = np.zeros((2, 2))

	for i, cl in enumerate(clip_limit):
		print(clip_limit)
		img = preprocess_image(image, clip_limit=cl, interval=guess[1], threshold=True)
		snr[0][i] = get_snr(img)
	for j, w in enumerate(sigma):
		img = preprocess_image(image, clip_limit=guess[0], interval=w, threshold=True)
		snr[1][j] = get_snr(img)

	print(snr)

	jacobian[0] = 0.5 * (snr[0][1] - snr[0][0]) / inc
	jacobian[1] = 0.5 * (snr[1][1] - snr[1][0]) / inc

	"""
	hessian[0][0] = 0.25 * (snr[2][1] - 2 * snr[1][1] - snr[0][1]) / inc
	hessian[0][1] = 0.25 * (snr[2][2] - snr[0][2] + snr[2][0] - snr[0][0]) / inc
	hessian[1][0] = 0.25 * (snr[2][2] - snr[2][0] + snr[0][2] - snr[0][0]) / inc
	hessian[1][1] = 0.25 * (snr[1][2] - 2 * snr[1][1] - snr[1][0]) / inc
	"""

	img = preprocess_image(image, clip_limit=guess[0], sigma=guess[1], threshold=True)
	snr = get_snr(img)

	return snr, jacobian, hessian

def get_snr(image):

	noise = estimate_sigma(image, multichannel=False, average_sigmas=True)
	signal = image.sum() / np.count_nonzero(image)

	return signal / noise


def optimise_equalisation(image, guess=[0.01, 0.9], alpha = 1.0, precision = 5E-6, max_it=100):

	iteration = 1
	x = np.array([guess, [guess[0]+0.0001, guess[1]+0.0001]])
	snr = []
	snr_grad = []

	snr_n, d_snr, dd_snr = get_snr_estimates(image, x[0])
	snr.append(snr_n)
	snr_grad.append(d_snr)

	snr_n, d_snr, dd_snr = get_snr_estimates(image, x[1])
	snr.append(snr_n)
	snr_grad.append(d_snr)

	print(snr, snr_grad)

	check = False
	while not check:
		gamma = d_snr * (x[-1] - x[-2]).T * (snr_grad[-1] - snr_grad[-2]) \
				/ ((snr_grad[-1] - snr_grad[-2])**2).sum()

		new_x = x[-1] + (alpha * gamma)

		check = np.all(new_x >= 0) * np.all(new_x < 1.0)
		alpha *= 0.9

		print(new_x, check, d_snr, gamma)

	x = np.concatenate((x, np.expand_dims(new_x, axis=0)))
	
	while True:
		snr_n, d_snr, dd_snr = get_snr_estimates(image, x[-1])
		snr.append(snr_n)
		snr_grad.append(d_snr)

		gamma = d_snr * (x[-1] - x[-2]).T * (snr_grad[-1] - snr_grad[-2]) \
				/ ((snr_grad[-1] - snr_grad[-2])**2).sum()

		new_x = x[-1] + gamma

		check = np.all(new_x >= 0) * np.all(new_x < 1.0) * \
			(iteration <= max_it) * (abs(gamma).sum() >= precision)

		print(new_x, check)
		print(iteration, d_snr, gamma)

		if not check:

			clip_limit = x[np.argmax(snr)][0]
			sigma = x[np.argmax(snr)][1]

			#"""
			import matplotlib
			matplotlib.use("TkAgg")
			import matplotlib.pyplot as plt
			
			X = np.sqrt((x**2).sum(axis=-1))

			plt.clf()
			plt.figure(0)
			plt.scatter(X[0], snr[0])
			plt.scatter(X[-1], snr[-1])
			plt.plot(X, snr)
			plt.savefig("gradient_ascent.png")
			
			eq_image = preprocess_image(image, threshold=True, clip_limit=clip_limit, sigma=sigma)

			plt.clf()
			plt.figure(0)
			plt.imshow(eq_image)
			plt.savefig("equalised_image.png")
			#"""
			return clip_limit, sigma, np.max(snr)

		x = np.concatenate((x, np.expand_dims(new_x, axis=0)))

		iteration += 1

"""Deprecated"""

def smart_nematic_tensor_analysis(nem_vector, precision=1E-1):
	"""
	nematic_tensor_analysis(nem_vector)

	Calculates eigenvalues and eigenvectors of average nematic tensor over area^2 pixels for n_samples

	Parameters
	----------

	nem_vector:  array_like (float); shape(n_frame, n_y, n_x, 4)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	av_eigval:  array_like (float); shape=(n_frame, n_sample, 2)
		Eigenvalues of average nematic tensors for n_sample areas

	av_eigvec:  array_like (float); shape=(n_frame, n_sample, 2, 2)
		Eigenvectors of average nematic tensors for n_sample areas

	"""

	tot_q, tot_angle, tot_energy = tensor_analysis(nem_vector)

	n_sample = nem_vector.shape[0]
	
	for n in range(n_sample):
		section = np.argmax(tot_q[n])

	def rec_search(nem_vector, q):

		image_shape = q.shape
		if image_shape[0] <= 2: return q

		for i in range(2):
			for j in range(2):
				vec_section = nem_vector[:,
									i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
									j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2 ]
				q_section = q[i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
							j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2]

				new_q, _ = tensor_analysis(np.mean(vector_map, axis=(1, 2)))
				old_q = np.mean(q_section)

				if abs(new_q - old_q) >= precision: q_section = rec_search(vec_section, q_section)
				else: q_section = np.ones(vec_section.shape[1:]) * new_q

				q[i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
				  j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2] = q_section

		return q

	for n in range(n_sample):
		vector_map = nem_vector[n]
		q0 = np.zeros(map_shape)

		print(vector_map.shape, np.mean(vector_map, axis=(1, 2)))

		av_q, _ = tensor_analysis(np.mean(vector_map, axis=(1, 2)))

		q0 += av_q
		q1 = rec_search(vector_map, q0)

		tot_q[n] = np.mean(np.unique(q1))

	return tot_q


def fourier_transform_analysis(image_shg):
	"""
	fourier_transform_analysis(image_shg, area, n_sample)

	Calculates fourier amplitude spectrum of over area^2 pixels for n_samples

	Parameters
	----------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	angles:  array_like (float); shape=(n_bins)
		Angles corresponding to fourier amplitudes

	fourier_spec:  array_like (float); shape=(n_bins)
		Average Fouier amplitudes of FT of image_shg

	"""

	n_sample = image_shg.shape[0]

	image_fft = np.fft.fft2(image_shg[0])
	image_fft[0][0] = 0
	image_fft = np.fft.fftshift(image_fft)
	average_fft = np.zeros(image_fft.shape, dtype=complex)

	fft_angle = np.angle(image_fft, deg=True)
	fft_freqs = np.fft.fftfreq(image_fft.size)
	angles = np.unique(fft_angle)
	fourier_spec = np.zeros(angles.shape)
	
	n_bins = fourier_spec.size

	for n in range(n_sample):
		image_fft = np.fft.fft2(image_shg[n])
		image_fft[0][0] = 0
		average_fft += np.fft.fftshift(image_fft) / n_sample	

	for i in range(n_bins):
		indices = np.where(fft_angle == angles[i])
		fourier_spec[i] += np.sum(np.abs(average_fft[indices])) / 360

	#A = np.sqrt(average_fft * fft_angle.size * fft_freqs**2 * (np.cos(fft_angle)**2 + np.sin(fft_angle)**2))

	sdi = np.mean(fourier_spec) / np.max(fourier_spec)

	return angles, fourier_spec, sdi
