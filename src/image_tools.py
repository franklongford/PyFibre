"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import sys, os, time
import numpy as np
import scipy as sp

import networkx as nx
from networkx.algorithms import approximation as approx
from networkx.algorithms.efficiency import local_efficiency, global_efficiency

from PIL import Image

from scipy.misc import derivative
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from scipy.optimize import curve_fit, minimize

from skimage import data, measure, img_as_float, draw, io
from skimage.transform import rescale
from skimage.morphology import (disk, dilation)
from skimage.color import rgb2grey, rgb2hsv, hsv2rgb
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.feature import structure_tensor, hessian_matrix
from skimage.exposure import rescale_intensity

import utilities as ut
from filters import tubeness
from extraction import FIRE, adj_analysis


def load_image(image_name):

	image_orig = io.imread(image_name).astype(float)

	if image_orig.ndim > 2: 
		if image_orig.ndim == 4: image_orig = image_orig[0]
		smallest_axis = np.argmin(image_orig.shape)
		image_orig = np.sqrt(np.sum(image_orig**2, axis=smallest_axis))

	image = image_orig / image_orig.max()

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


def preprocess_image(image, scale=1, p_intensity=(1, 98), p_denoise=(12, 35)):
	"""
	Pre-process image to remove outliers, reduce noise and rescale

	Parameters
	----------

	image:  array_like (float); shape=(n_y, n_x)
		Image to pre-process

	scale:  float
		Metric indicating image rescaling required

	p_intensity: tuple (float); shape=(2,)
		Percentile range for intensity rescaling (used to remove outliers)
	
	p_denoise: tuple (float); shape=(2,)
		Parameters for non-linear means denoise algorithm (used to remove noise)

	Returns
	-------

	image:  array_like (float); shape=(n_y, n_x)
		Pre-processed image

	"""

	low, high = np.percentile(image, p_intensity)
	image = rescale_intensity(image, in_range=(low, high))

	sigma = estimate_sigma(image)
	image = denoise_nl_means(image, patch_size=p_denoise[0], patch_distance=p_denoise[1],
				fast_mode=True, h = 1.2 * sigma, sigma=sigma, multichannel=False)

	image = rescale(image, scale)
	image /= image.max()

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

    if sigma != None: image = filters.gaussian_filter(image, sigma)
    
    image_fft = np.fft.fft2(image)
    image_ifft =  np.fft.ifft2(image_fft)
    
    fft_angle = np.angle(image_fft, deg=True)
    fft_magnitude = np.where(fft_angle == 0, 0, np.abs(image_fft))
    fft_order = np.argsort(fft_angle.flatten())
    
    sdi = np.mean(fft_magnitude) / np.max(fft_magnitude)

    return fft_angle.flatten()[fft_order], fft_magnitude.flatten()[fft_order], sdi


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


def network_extraction(image, network_name='network', sigma=1.0, scale=1,
				p_intensity=(1, 98), p_denoise=(12, 35), ow_network=False, threads=8):
	"""
	Extract fibre network using modified FIRE algorithm
	"""

	"Try loading saved graph opbject"
	try: Aij = nx.read_gpickle(network_name + "_network.pkl")
	except IOError: ow_network = True

	"Else, use modified FIRE algorithm to extract network"
	if ow_network:
		"Apply tubeness transform to enhance image fibres"
		image_TB = tubeness(image, sigma)
		"Call FIRE algorithm to extract full image network"
		Aij = FIRE(image_TB, sigma=sigma, max_threads=threads)
		nx.write_gpickle(Aij, network_name + "_network.pkl")

	label_image = np.zeros(image.shape)
	networks = []

	"Segment image based on connected components in network"
	for i, component in enumerate(nx.connected_components(Aij)):
		subgraph = Aij.subgraph(component)
		if subgraph.number_of_nodes() > 3:
			networks.append(subgraph)
			label_image = draw_network(subgraph, label_image, i + 1)

	n_clusters = len(networks)
	label_image = np.array(label_image, dtype=int)
	segmented_image = np.zeros(image.shape, dtype=int)
	areas = np.empty((0,), dtype=float)
	regions = []
	segments = []

	"Measure pixel areas of each segment"
	for region in measure.regionprops(label_image):
		minr, minc, maxr, maxc = region.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		area, segment = network_area(region.image)

		if region.area >= 4E-4 * image.size:
			segmented_image[(indices[0], indices[1])] += segment * region.label
			areas = np.concatenate((areas, [area]))
			regions.append(region)
			segments.append(segment)

	"Sort segments ranked by area"
	indices = np.argsort(areas)
	sorted_areas = areas[indices]
	sorted_regions = []
	sorted_segments = []
	sorted_networks = []

	for index in indices: 
		sorted_regions.append(regions[index])
		sorted_segments.append(segments[index])
		sorted_networks.append(networks[index])

	return segmented_image, sorted_networks, sorted_areas, sorted_regions, sorted_segments


def draw_network(network, label_image, index):

	nodes_coord = [network.nodes[i]['xy'] for i in network.nodes()]
	nodes_coord = np.stack(nodes_coord)
	label_image[nodes_coord[:,0],nodes_coord[:,1]] = index

	for edge in list(network.edges):
		start = list(network.nodes[edge[1]]['xy'])
		end = list(network.nodes[edge[0]]['xy'])
		line = draw.line(*(start+end))
		label_image[line] = index

	return label_image


def network_area(label, iterations=6):

	dilated_image = binary_dilation(label, iterations=iterations)
	filled_image = dilated_image#binary_fill_holes(dilated_image)
	net_area = np.sum(filled_image)

	return net_area, filled_image


def network_analysis(image, networks, regions, segments, n_tensor, anis_map):
	"""
	Analyse extracted fibre network
	"""
	l_regions = len(regions)

	region_sdi = np.zeros(l_regions)
	region_anis = np.zeros(l_regions)
	region_pix_anis = np.zeros(l_regions)

	segment_linear = np.zeros(l_regions)
	segment_eccent = np.zeros(l_regions)
	segment_density = np.zeros(l_regions)
	segment_coverage = np.zeros(l_regions)
	
	network_waviness = np.zeros(l_regions)
	network_degree = np.zeros(l_regions)
	network_central = np.zeros(l_regions)
	network_connect = np.zeros(l_regions)
	network_loc_eff = np.zeros(l_regions)
	network_glob_eff = np.zeros(l_regions)

	waviness_time = 0
	degree_time = 0
	central_time = 0
	connect_time = 0
	loc_eff_time = 0
	glob_eff_time = 0

	iterator = zip(np.arange(l_regions), networks, regions, segments)

	for i, network, region, segment in iterator:

		minr, minc, maxr, maxc = region.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]
		region_image = image[(indices[0], indices[1])]

		_, _, sdi = fourier_transform_analysis(region_image)
		region_sdi[i] = sdi

		region_anis_map = anis_map[(indices[0], indices[1])]
		region_n_tensor = n_tensor[(indices[0], indices[1])]

		region_anis[i], _ , _ = tensor_analysis(np.mean(region_n_tensor, axis=(0, 1)))
		region_pix_anis[i] = np.mean(region_anis_map)

		filter_ = np.where(segment, 1, 0)
		seg_region = measure.regionprops(filter_)[0]

		segment_linear[i] += 1 - seg_region.equivalent_diameter / seg_region.perimeter
		segment_eccent[i] += seg_region.eccentricity
		segment_density[i] += np.sum(region_image * filter_) / np.sum(filter_)
		segment_coverage[i] += np.mean(filter_)

		print(segment.size, image.size)

		start = time.time()
		network_waviness[i] = adj_analysis(network)
		stop1 = time.time()
		waviness_time += stop1-start

		try: network_degree[i] = nx.degree_pearson_correlation_coefficient(network)**2
		except: network_degree[i] = None
		stop2 = time.time()
		degree_time += stop2-stop1

		try: network_central[i] = np.real(nx.adjacency_spectrum(network).max())
		except: network_central[i] = None
		stop3 = time.time()
		central_time += stop3-stop2

		try: network_connect[i] = nx.algebraic_connectivity(network)
		except: network_connect[i] = None
		stop4 = time.time()
		connect_time += stop4-stop3

		try: network_loc_eff[i] = local_efficiency(network)
		except: network_loc_eff[i] = None
		stop5 = time.time()
		loc_eff_time += stop5-stop4

	#print('Network Waviness = {} s'.format(waviness_time))
	#print('Network Degree = {} s'.format(degree_time))
	#print('Network Centrality = {} s'.format(central_time))
	#print('Network Conectivity = {} s'.format(connect_time))
	#print('Network Local Efficiency = {} s'.format(loc_eff_time))

	return (region_sdi, region_anis, region_pix_anis, 
			segment_linear, segment_eccent, segment_density, segment_coverage,
			network_waviness, network_degree, network_central, 
			network_connect, network_loc_eff)

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
