"""
PyFibre
Image Segmentation Library 

Created by: Frank Longford
Created on: 18/02/2019

Last Modified: 18/02/2019
"""

import sys, os, time
import numpy as np
import scipy as sp

import networkx as nx
from networkx.algorithms import cluster
from networkx.algorithms import approximation as approx
from networkx.algorithms.efficiency import local_efficiency, global_efficiency

import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_closing, binary_opening

from skimage import measure, draw
from skimage.util import pad
from skimage.transform import rescale, resize
from skimage.feature import greycomatrix
from skimage.morphology import remove_small_objects, remove_small_holes, dilation, disk
from skimage.color import grey2rgb, rgb2grey
from skimage.filters import threshold_otsu, threshold_isodata, threshold_mean, rank, apply_hysteresis_threshold
from skimage.exposure import rescale_intensity, equalize_hist, equalize_adapthist

from sklearn.cluster import MiniBatchKMeans

import utilities as ut
from filters import tubeness, hysteresis
from extraction import FIRE, fibre_assignment, simplify_network
from analysis import (fourier_transform_analysis, tensor_analysis, angle_analysis, 
					fibre_analysis, greycoprops_edit)
from preprocessing import nl_means, clip_intensities


def create_binary_image(segments, shape):

	binary_image = np.zeros(shape)

	for segment in segments:
		minr, minc, maxr, maxc = segment.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]
		binary_image[(indices[0], indices[1])] += segment.image

	binary_image = np.where(binary_image, 1, 0)

	return binary_image


def segment_check(segment, min_size=0, min_frac=0, edges=False, max_x=0, max_y=0):

	segment_check = True
	minr, minc, maxr, maxc = segment.bbox

	if edges:
		edge_check = (minr != 0) * (minc != 0)
		edge_check *= (maxr != max_x)
		edge_check *= (maxc != max_y)

		segment_check *= edge_check

	segment_check *= segment.filled_area >= min_size
	segment_frac = (segment.image * segment.intensity_image).sum() / segment.filled_area
	segment_check *= (segment_frac >= min_frac)

	return segment_check

def get_segments(image, binary, min_size=0, min_frac=0):

	labels = measure.label(binary.astype(np.int))
	segments = []
	areas = []

	for segment in measure.regionprops(labels, intensity_image=image, coordinates='xy'):
		seg_check = segment_check(segment, min_size, min_frac)

		if seg_check:
			segments.append(segment)
			areas.append(segment.filled_area)

	indices = np.argsort(areas)[::-1]
	sorted_segs = [segments[i] for i in indices]

	return sorted_segs


def prepare_composite_image(image, p_intensity=(2, 98), sm_size=7):

	image_size = image.shape[0] * image.shape[1]
	image_shape = (image.shape[0], image.shape[1])
	image_channels = image.shape[-1]
	image_scaled = np.zeros(image.shape, dtype=int)
	pad_size = 10 * sm_size

	"Mimic contrast stretching decorrstrech routine in MatLab"
	for i in range(image_channels):
		image_scaled[:, :, i] = 255 * clip_intensities(image[:, :, i], p_intensity=p_intensity)

	"Pad each channel, equalise and smooth to remove salt and pepper noise"
	for i in range(image_channels):
		padded = pad(image_scaled[:, :, i], [pad_size, pad_size], 'symmetric')
		equalised = 255 * equalize_hist(padded)
		smoothed = median_filter(equalised, size=(sm_size, sm_size))
		smoothed = median_filter(smoothed, size=(sm_size, sm_size))
		image_scaled[:, :, i] = smoothed[pad_size : pad_size + image.shape[0],
						 	pad_size : pad_size + image.shape[1]]

	return image_scaled


def cluster_colours(image, n_clusters=8, n_init=10):

	image_size = image.shape[0] * image.shape[1]
	image_shape = (image.shape[0], image.shape[1])
	image_channels = image.shape[-1]

	"Perform k-means clustering on PL image"
	X = np.array(image.reshape((image_size, image_channels)), dtype=float)
	clustering = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init, 
				    reassignment_ratio=0.99, init_size=n_init*100,
				    max_no_improvement=15)
	clustering.fit(X)

	labels = clustering.labels_.reshape(image_shape)
	centres = clustering.cluster_centers_

	return labels, centres


def BD_filter(image, n_runs=1, n_clusters=10, p_intensity=(2, 98), sm_size=7, param=[0.7, 1.1, 1.40, 0.82]):
	"Adapted from CurveAlign BDcreationHE routine"

	assert image.ndim == 3

	image_size = image.shape[0] * image.shape[1]
	image_shape = (image.shape[0], image.shape[1])
	image_channels = image.shape[-1]

	image_scaled = prepare_composite_image(image, p_intensity, sm_size)

	print("Making greyscale")
	greyscale = rgb2grey(image_scaled.astype(np.float64))
	greyscale /= greyscale.max()

	tot_labels = []
	tot_centres = []
	tot_cell_clusters = []
	cost_func = np.zeros(n_runs)

	for run in range(n_runs):

		labels, centres = cluster_colours(image_scaled, n_clusters)
		tot_labels.append(labels)

		"Reorder labels to represent average intensity"
		intensities = np.zeros(n_clusters)

		for i in range(n_clusters):
			intensities[i] = greyscale[np.where(labels == i)].sum() / np.where(labels == i, 1, 0).sum()

		magnitudes = np.sqrt(np.sum(centres**2, axis=-1))
		norm_centres = centres / np.repeat(magnitudes, image_channels).reshape(centres.shape)
		tot_centres.append(norm_centres)

		"Convert RGB centroids to spherical coordinates"
		X = np.arcsin(norm_centres[:, 0])
		Y = np.arcsin(norm_centres[:, 1])
		Z = np.arccos(norm_centres[:, 2])
		I = intensities

		"Define the plane of division between cellular and fibourus clusters"	
		cell_clusters = (X <= param[0]) * (Y <= param[1]) * (Z <= param[2]) * (I <= param[3])
		chosen_clusters = np.argwhere(cell_clusters).flatten()
		cost_func[run] += X[chosen_clusters].mean() +  Y[chosen_clusters].mean() \
				 + Z[chosen_clusters].mean() + I[chosen_clusters].mean()
		tot_cell_clusters.append(chosen_clusters)

	labels = tot_labels[cost_func.argmin()]
	norm_centres = tot_centres[cost_func.argmin()]
	cell_clusters = tot_cell_clusters[cost_func.argmin()]

	intensities = np.zeros(n_clusters)
	segmented_image = np.zeros((n_clusters,) + image.shape, dtype=int)
	for i in range(n_clusters):
		segmented_image[i][np.where(labels == i)] += image_scaled[np.where(labels == i)]
		intensities[i] = greyscale[np.where(labels == i)].sum() / np.where(labels == i, 1, 0).sum()

	"Select blue regions to extract epithelial cells"
	epith_cell = np.zeros(image.shape)
	for i in cell_clusters: epith_cell += segmented_image[i]
	epith_grey = rgb2grey(epith_cell)

	"Convert RGB centroids to spherical coordinates"
	X = np.arcsin(norm_centres[:, 0])
	Y = np.arcsin(norm_centres[:, 1])
	Z = np.arccos(norm_centres[:, 2])
	I = intensities

	"""
	print(X, Y, Z, I)
	print((X <= param[0]) * (Y <= param[1]) * (Z <= param[2]) * (I <= param[3]))

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	plt.figure(100, figsize=(10, 10))
	plt.imshow(image)
	plt.axis('off')

	plt.figure(1000, figsize=(10, 10))
	plt.imshow(image_scaled)
	plt.axis('off')

	for i in range(n_clusters):
		plt.figure(i)
		plt.imshow(segmented_image[i])

	not_clusters = [i for i in range(n_clusters) if i not in cell_clusters]

	plt.figure(1001)
	plt.scatter(X[cell_clusters], Y[cell_clusters])
	plt.scatter(X[not_clusters], Y[not_clusters])
	for i in range(n_clusters): plt.annotate(i, (X[i], Y[i]))

	plt.figure(1002)
	plt.scatter(X[cell_clusters], Z[cell_clusters])
	plt.scatter(X[not_clusters], Z[not_clusters])
	for i in range(n_clusters): plt.annotate(i, (X[i], Z[i]))

	plt.figure(1003)
	plt.scatter(X[cell_clusters], I[cell_clusters])
	plt.scatter(X[not_clusters], I[not_clusters])
	for i in range(n_clusters): plt.annotate(i, (X[i], I[i]))

	plt.show()	
	#"""

	"Dilate binary image to smooth regions and remove small holes / objects"
	epith_cell_BW = np.where(epith_grey, True, False)
	epith_cell_BW_open = binary_opening(epith_cell_BW, iterations=2)

	BWx = binary_fill_holes(epith_cell_BW_open)
	BWy = remove_small_objects(~BWx, min_size=20)

	"Return binary mask for cell identification"
	mask_image = remove_small_objects(~BWy, min_size=20)

	return mask_image



def cell_segmentation(image_shg, image_pl, image_tran, scale=1.0, sigma=0.8, alpha=1.0,
			min_size=400, edges=False):
	"Return binary filter for cellular identification"


	min_size *= scale**2

	image_shg = np.sqrt(image_shg * image_tran)
	image_pl = np.sqrt(image_pl * image_tran)
	image_tran = equalize_adapthist(image_tran)

	"Create composite RGB image from SHG, PL and transmission"
	image_stack = np.stack((image_shg, image_pl, image_tran), axis=-1)
	magnitudes = np.sqrt(np.sum(image_stack**2, axis=-1))
	indices = np.nonzero(magnitudes)
	image_stack[indices] /= np.repeat(magnitudes[indices], 3).reshape(indices[0].shape + (3,))

	"Up-scale image to impove accuracy of clustering"
	print(f"Rescaling by {scale}")
	image_stack = rescale(image_stack, scale, multichannel=True, mode='constant', anti_aliasing=None)

	"Form mask using Kmeans Background filter"
	mask_image = BD_filter(image_stack)
	print("Resizing")
	mask_image = resize(mask_image, image_shg.shape, mode='reflect', anti_aliasing=True)

	cells = []
	cell_areas = []
	fibres = []
	fibre_areas = []

	cell_binary = np.array(mask_image, dtype=bool)
	fibre_binary = np.where(mask_image, False, True)

	cell_labels = measure.label(cell_binary.astype(np.int))
	for cell in measure.regionprops(cell_labels, intensity_image=image_pl, coordinates='xy'):
		cell_check = segment_check(cell, min_size, 0.01)

		if not cell_check:
			minr, minc, maxr, maxc = cell.bbox
			indices = np.mgrid[minr:maxr, minc:maxc]
			cell_binary[(indices[0], indices[1])] = False

			fibre = measure.regionprops(np.array(cell.image, dtype=int),
							intensity_image=image_shg[(indices[0], indices[1])],
							coordinates='xy')[0]

			fibre_check = segment_check(fibre, 0, 0.075)
			if fibre_check: fibre_binary[(indices[0], indices[1])] = True

	print("Removing small holes")
	fibre_binary = remove_small_holes(fibre_binary)
	cell_binary = remove_small_holes(cell_binary)

	sorted_fibres = get_segments(image_shg, fibre_binary, min_size, 0.075)
	sorted_cells = get_segments(image_pl, cell_binary, min_size, 0.01)

	return sorted_cells, sorted_fibres


def hysteresis_binary(image, segments_low, segments_high, iterations=1, min_size=0, min_frac=0):

	image = equalize_adapthist(image)

	binary_low = create_binary_image(segments_low, image.shape)
	binary_high = create_binary_image(segments_high, image.shape)

	binary_high = binary_dilation(binary_high, iterations=1)
	binary_high = binary_closing(binary_high)

	intensity_map_low = image * binary_low
	intensity_map_high = image * binary_high

	intensity_map = 0.5 * (intensity_map_low + intensity_map_high)
	intensity_binary = np.where(intensity_map >= 0.15, True, False)

	intensity_binary = remove_small_holes(intensity_binary)
	intensity_binary = remove_small_objects(intensity_binary)
	thresholded = binary_dilation(intensity_binary, iterations=1)

	"""
	labels_low, num_labels = ndi.label(binary_low)
	# Check which connected components contain pixels from mask_high
	sums = ndi.sum(binary_high, labels_low, np.arange(num_labels + 1))

	connected_to_high = sums > 0
	connected_to_high[0] = False

	thresholded = connected_to_high[labels_low]
	#"""
	
	smoothed = gaussian_filter(thresholded.astype(np.float), sigma=1.5)
	smoothed = np.where(smoothed >= 0.75, True, False)

	"""
	plt.figure(0)
	plt.imshow(intensity_map)
	plt.figure(1)
	plt.imshow(thresholded)
	plt.figure(2)
	plt.imshow(smoothed)
	plt.show()
	"""

	return smoothed


def cell_segment_analysis(image, cells, n_tensor, anis_map, angle_map):

	l_cells = len(cells)

	cell_angle_sdi = np.zeros(l_cells)
	cell_anis = np.zeros(l_cells)
	cell_pix_anis = np.zeros(l_cells)

	cell_areas = np.zeros(l_cells)
	cell_hu = np.zeros((l_cells, 7))
	cell_mean = np.zeros(l_cells)
	cell_std = np.zeros(l_cells)
	cell_entropy = np.zeros(l_cells)

	cell_glcm_contrast = np.zeros(l_cells)
	cell_glcm_dissim = np.zeros(l_cells)
	cell_glcm_corr = np.zeros(l_cells)
	cell_glcm_homo = np.zeros(l_cells)
	cell_glcm_energy = np.zeros(l_cells)
	cell_glcm_IDM = np.zeros(l_cells)
	cell_glcm_variance = np.zeros(l_cells)
	cell_glcm_cluster = np.zeros(l_cells)
	cell_glcm_entropy = np.zeros(l_cells)

	cell_linear = np.zeros(l_cells)
	cell_eccent = np.zeros(l_cells)
	cell_density = np.zeros(l_cells)
	cell_coverage = np.zeros(l_cells)

	for i, cell in enumerate(cells):

		metrics = segment_analysis(image, cell, n_tensor, anis_map,
						angle_map)

		(__, cell_angle_sdi[i], cell_anis[i], cell_pix_anis[i], 
		cell_areas[i], cell_linear[i], cell_eccent[i], 
		cell_density[i], cell_coverage[i], cell_mean[i], cell_std[i],
		cell_entropy[i], cell_glcm_contrast[i], cell_glcm_homo[i], cell_glcm_dissim[i], 
		cell_glcm_corr[i], cell_glcm_energy[i], cell_glcm_IDM[i], 
		cell_glcm_variance[i], cell_glcm_cluster[i], cell_glcm_entropy[i],
		cell_hu[i]) = metrics

	return (cell_angle_sdi, cell_anis, cell_pix_anis, cell_areas, 
		cell_mean, cell_std, cell_entropy, cell_glcm_contrast,
		cell_glcm_homo, cell_glcm_dissim, cell_glcm_corr, cell_glcm_energy,
		cell_glcm_IDM, cell_glcm_variance, cell_glcm_cluster, cell_glcm_entropy,
		cell_linear, cell_eccent, cell_density, cell_coverage, cell_hu) 


def segment_analysis(image, segment, n_tensor, anis_map, angle_map):

	minr, minc, maxr, maxc = segment.bbox
	indices = np.mgrid[minr:maxr, minc:maxc]

	segment_image = image[(indices[0], indices[1])]
	#segment_image_comb = np.sqrt(segment_image_shg * segment_image_pl)
	segment_anis_map = anis_map[(indices[0], indices[1])]
	segment_angle_map = angle_map[(indices[0], indices[1])]
	segment_n_tensor = n_tensor[(indices[0], indices[1])]

	_, _, segment_fourier_sdi = fourier_transform_analysis(segment_image)
	segment_angle_sdi = angle_analysis(segment_angle_map, segment_anis_map)

	segment_mean = np.mean(segment_image)
	segment_std = np.std(segment_image)
	segment_entropy = measure.shannon_entropy(segment_image)

	segment_anis, _ , _ = tensor_analysis(np.mean(segment_n_tensor, axis=(0, 1)))
	segment_anis = segment_anis[0]
	segment_pix_anis = np.mean(segment_anis_map)

	segment_area = np.sum(segment.image)
	segment_linear = 1 - segment.equivalent_diameter / segment.perimeter
	segment_eccent = segment.eccentricity
	segment_density = np.sum(segment_image * segment.image) / segment_area
	segment_coverage = np.mean(segment.image)
	segment_hu = segment.moments_hu

	glcm = greycomatrix((segment_image * segment.image * 255.999).astype('uint8'),
                         [1, 2], [0, np.pi/4, np.pi/2, np.pi*3/4], 256,
                         symmetric=True, normed=True)

	segment_glcm_contrast = greycoprops_edit(glcm, 'contrast').mean()
	segment_glcm_homo = greycoprops_edit(glcm, 'homogeneity').mean()
	segment_glcm_dissim = greycoprops_edit(glcm, 'dissimilarity').mean()
	segment_glcm_corr = greycoprops_edit(glcm, 'correlation').mean()
	segment_glcm_energy = greycoprops_edit(glcm, 'energy').mean()
	segment_glcm_IDM = greycoprops_edit(glcm, 'IDM').mean()
	segment_glcm_variance = greycoprops_edit(glcm, 'variance').mean()
	segment_glcm_cluster = greycoprops_edit(glcm, 'cluster').mean()
	segment_glcm_entropy = greycoprops_edit(glcm, 'entropy').mean()


	return (segment_fourier_sdi, segment_angle_sdi, segment_anis, segment_pix_anis, 
		segment_area, segment_linear, segment_eccent, segment_density, 
		segment_coverage, segment_mean, segment_std, segment_entropy,
		segment_glcm_contrast, segment_glcm_homo, segment_glcm_dissim, 
		segment_glcm_corr, segment_glcm_energy, segment_glcm_IDM, 
		segment_glcm_variance, segment_glcm_cluster, segment_glcm_entropy,
		segment_hu)


def network_extraction(image_shg, network_name='network', scale=1.0, sigma=0.75, alpha=0.5,
			p_denoise=(5, 35), threads=8):
	"""
	Extract fibre network using modified FIRE algorithm
	"""

	print("Applying AHE to SHG image")
	image_shg = equalize_adapthist(image_shg)
	print("Performing NL Denoise using local windows {} {}".format(*p_denoise))
	image_nl = nl_means(image_shg, p_denoise=p_denoise)

	"Call FIRE algorithm to extract full image network"
	print("Calling FIRE algorithm using image scale {}  alpha  {}".format(scale, alpha))
	Aij = FIRE(image_nl, scale=scale, sigma=sigma, alpha=alpha, max_threads=threads)
	nx.write_gpickle(Aij, network_name + "_graph.pkl")

	print("Extracting and simplifying fibre networks from graph")
	n_nodes = []
	networks = []
	networks_red = []
	fibres = []
	for i, component in enumerate(nx.connected_components(Aij)):
		subgraph = Aij.subgraph(component)

		fibre = fibre_assignment(subgraph)

		if len(fibre) > 0:
			n_nodes.append(subgraph.number_of_nodes())
			networks.append(subgraph)
			networks_red.append(simplify_network(subgraph))
			fibres.append(fibre)		

	"Sort segments ranked by network size"
	indices = np.argsort(n_nodes)[::-1]

	sorted_networks = [networks[i] for i in indices]
	sorted_networks_red = [networks_red[i] for i in indices]
	sorted_fibres = [fibres[i] for i in indices]

	return sorted_networks, sorted_networks_red, sorted_fibres


def fibre_segmentation(image_shg, networks, networks_red, area_threshold=200, iterations=8):

	n_net = len(networks)
	fibres = []

	iterator = zip(np.arange(n_net), networks, networks_red)

	"Segment image based on connected components in network"
	for i, network, network_red in iterator:

		label_image = np.zeros(image_shg.shape, dtype=int)
		label_image = draw_network(network, label_image, 1)

		dilated_image = binary_dilation(label_image, iterations=iterations)
		smoothed_image = gaussian_filter(dilated_image, sigma=0.5)
		filled_image = remove_small_holes(smoothed_image, area_threshold=area_threshold)
		binary_image = np.where(filled_image > 0, 1, 0)

		segment = measure.regionprops(binary_image, intensity_image=image_shg, coordinates='xy')[0]
		area = np.sum(segment.image)

		minr, minc, maxr, maxc = segment.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		#print(network.number_of_nodes(), area, 1E-2 * image_shg.size)

		segment.label = (i + 1)
		fibres.append(segment)

	return fibres


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


def filter_segments(segments, network, network_red, min_size=200):

	remove_list = []
	for i, segment in enumerate(segments):
		area = np.sum(segment.image)
		if area < min_size: remove_list.append(i)
			
	for i in remove_list:
		segments.remove(segment[i])
		network.remove(network[i])
		network_red.remove(network_red[i])
	
	return 	segments, network, network_red
		 


def network_analysis(network, network_red):

	cross_links = np.array([degree[1] for degree in network.degree], dtype=int)
	network_cross_links = (cross_links > 2).sum()

	try: network_degree = nx.degree_pearson_correlation_coefficient(network, weight='r')**2
	except: network_degree = None

	try: network_eigen = np.real(nx.adjacency_spectrum(network_red).max())
	except: network_eigen = None

	try: network_connect = nx.algebraic_connectivity(network_red, weight='r')
	except: network_connect = None

	"""
	try: network_loc_eff = local_efficiency(network_red)
	except: network_loc_eff = None

	try: network_cluster = cluster.average_clustering(network_red, weight='r')
	except: network_cluster = None
	"""

	return (network_degree, network_eigen, network_connect, network_cross_links)




def fibre_segment_analysis(image_shg, networks, networks_red, 
			fibres, segments, n_tensor, anis_map, angle_map):
	"""
	Analyse extracted fibre network
	"""
	l_regions = len(segments)

	segment_fourier_sdi = np.zeros(l_regions)
	segment_angle_sdi = np.zeros(l_regions)
	segment_anis = np.zeros(l_regions)
	segment_pix_anis = np.zeros(l_regions)

	segment_area = np.zeros(l_regions)
	segment_linear = np.zeros(l_regions)
	segment_eccent = np.zeros(l_regions)
	segment_density = np.zeros(l_regions)
	segment_coverage = np.zeros(l_regions)

	segment_mean = np.zeros(l_regions)
	segment_std = np.zeros(l_regions)
	segment_entropy = np.zeros(l_regions)

	segment_glcm_contrast = np.zeros(l_regions)
	segment_glcm_dissim = np.zeros(l_regions)
	segment_glcm_corr = np.zeros(l_regions)
	segment_glcm_homo = np.zeros(l_regions)
	segment_glcm_energy = np.zeros(l_regions)
	segment_glcm_IDM = np.zeros(l_regions)
	segment_glcm_variance = np.zeros(l_regions)
	segment_glcm_cluster = np.zeros(l_regions)
	segment_glcm_entropy = np.zeros(l_regions)

	segment_hu = np.zeros((l_regions, 7))
	
	fibre_waviness = np.zeros(l_regions)
	fibre_lengths = np.zeros(l_regions)
	fibre_cross_link_den = np.zeros(l_regions)
	fibre_angle_sdi = np.zeros(l_regions)

	network_degree = np.zeros(l_regions)
	network_eigen = np.zeros(l_regions)
	network_connect = np.zeros(l_regions)

	iterator = zip(np.arange(l_regions), networks, networks_red, fibres, segments)

	for i, network, network_red, fibre, segment in iterator:

		#if segment.filled_area >= 1E-2 * image_shg.size:

		metrics = segment_analysis(image_shg, segment, n_tensor, anis_map,
						angle_map)

		(segment_fourier_sdi[i], segment_angle_sdi[i], segment_anis[i], 
		segment_pix_anis[i], segment_area[i], segment_linear[i], segment_eccent[i], 
		segment_density[i], segment_coverage[i], segment_mean[i], segment_std[i],
		segment_entropy[i], segment_glcm_contrast[i], segment_glcm_homo[i], segment_glcm_dissim[i], 
		segment_glcm_corr[i], segment_glcm_energy[i], segment_glcm_IDM[i], 
		segment_glcm_variance[i], segment_glcm_cluster[i], segment_glcm_entropy[i],
		segment_hu[i]) = metrics

		metrics = network_analysis(network, network_red)

		(network_degree[i], network_eigen[i],
		network_connect[i], network_cross_links) = metrics

		fibre_len, fibre_wav, fibre_ang = fibre_analysis(fibre)

		fibre_waviness[i] = np.nanmean(fibre_wav)
		fibre_lengths[i] = np.nanmean(fibre_len)
		fibre_cross_link_den[i] = network_cross_links / len(fibre)
		#fibre_angle_sdi[i] = angle_analysis(fibre_ang, np.ones(fibre_ang.shape))


	return (segment_angle_sdi, segment_anis, segment_pix_anis, 
		segment_area, segment_linear, segment_eccent, segment_density, segment_coverage,
		segment_mean, segment_std, segment_entropy, segment_glcm_contrast, 
		segment_glcm_homo, segment_glcm_dissim, segment_glcm_corr, segment_glcm_energy, 
		segment_glcm_IDM, segment_glcm_variance, segment_glcm_cluster, segment_glcm_entropy,
		segment_hu, network_degree, network_eigen, network_connect, fibre_waviness, 
		fibre_lengths, fibre_cross_link_den)
