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

from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_closing, binary_opening

from skimage import measure, draw
from skimage.util import pad
from skimage.transform import rescale, resize
from skimage.feature import greycomatrix
from skimage.morphology import remove_small_objects, remove_small_holes, dilation
from skimage.color import grey2rgb, rgb2grey
from skimage.filters import threshold_otsu, threshold_mean
from skimage.exposure import rescale_intensity, equalize_hist

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


def find_holes(image, sigma=0.8, alpha=1.0, min_size=1250, iterations=2):

	image_TB = tubeness(image, sigma=sigma)

	image_hyst = hysteresis(image_TB, alpha=alpha)
	image_hyst = remove_small_objects(image_hyst)
	image_hyst = binary_closing(image_hyst, iterations=iterations)

	image_hole = remove_small_holes(~image_hyst, min_size=min_size)
	image_hole = binary_opening(image_hole, iterations=iterations)
	image_hole = binary_fill_holes(image_hole)
	
	return image_hole


def BD_filter(image, n_runs=50, n_clusters=6, p_intensity=(2, 98), sm_size=7):
	"Adapted from CurveAlign BDcreationHE routine"

	assert image.ndim == 3

	image_size = image.shape[0] * image.shape[1]
	image_shape = (image.shape[0], image.shape[1])
	image_channels = image.shape[-1]
	image_scaled = np.zeros(image.shape, dtype=int)
	pad_size = 10 * sm_size

	"Mimic contrast stretching decorrstrech routine in MatLab"
	for i in range(image_channels):
		low, high = np.percentile(image[:, :, i], p_intensity) 
		image_scaled[:, :, i] = 255 * rescale_intensity(image[:, :, i], in_range=(low, high))

	"Pad each channel, equalise and smooth to remove salt and pepper noise"
	for i in range(image_channels):
		padded = pad(image_scaled[:, :, i], [pad_size, pad_size], 'symmetric')
		equalised = 255 * equalize_hist(padded)
		smoothed = median_filter(equalised, size=(sm_size, sm_size))
		smoothed = median_filter(smoothed, size=(sm_size, sm_size))
		image_scaled[:, :, i] = smoothed[pad_size : pad_size + image.shape[0],
						 pad_size : pad_size + image.shape[1]]

	"Perform k-means clustering on PL image"
	X = np.array(image_scaled.reshape((image_size, image_channels)), dtype=float)
	clustering = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_runs, 
				    reassignment_ratio=0.99, init_size=n_runs*100,
				    max_no_improvement=10)
	clustering.fit(X)

	labels = clustering.labels_.reshape(image_shape)
	centres = clustering.cluster_centers_

	"Reorder labels to represent average intensity"
	unique_labels = np.unique(labels)
	segmented_image = np.zeros((n_clusters,) + image.shape)
	mean_intensity = np.zeros(n_clusters)
	mean_intensity_vec = np.zeros((n_clusters, 3))

	for i in range(n_clusters):
		segmented_image[i][np.where(labels == i)] += 255 * image[np.where(labels == i)]
		grey = rgb2grey(segmented_image[i])
		mean_intensity[i] += np.mean(grey[np.nonzero(grey)])
		mean_intensity_vec[i] += segmented_image[i].sum(axis=(0,1))
		mean_intensity_vec[i] /= np.where(segmented_image[i], 1, 0).sum(axis=(0, 1))

	magnitudes = np.sqrt(np.sum(centres**2, axis=-1))
	norm_centres = centres / np.repeat(magnitudes, image_channels).reshape(centres.shape)

	magnitudes = np.sqrt(np.sum(mean_intensity_vec**2, axis=-1))
	norm_intensities = mean_intensity_vec / np.repeat(magnitudes, image_channels).reshape(mean_intensity_vec.shape)

	"""Light blue clusters classed as where kmeans centres have highest value in 
	B channel (index 2) and average normalised channel intensities below 0.92"""
	blue_clusters = np.array([vector[2] >= 0.4 for vector in norm_centres], dtype=bool)
	light_clusters = np.array([vector[2] >= 0.92 for vector in norm_intensities], dtype=bool)
	light_blue_clusters = np.argwhere(light_clusters).flatten()

	"Select blue regions to extract epithelial cells"
	epith_cell = np.zeros(image.shape)
	for i in light_blue_clusters: epith_cell += segmented_image[i]
	epith_grey = rgb2grey(epith_cell)

	"Dilate binary image to smooth regions and remove small holes / objects"
	epith_cell_BW = np.where(epith_grey, True, False)
	epith_cell_BW_open = binary_opening(epith_cell_BW, iterations=2)

	BWx = binary_fill_holes(epith_cell_BW_open)
	BWy = remove_small_objects(~BWx, min_size=15)

	"Return binary mask for cell identification"
	mask_image = remove_small_objects(~BWy, min_size=15)

	"""
	import matplotlib.pyplot as plt
	for i in range(n_clusters):
		plt.figure(i)
		plt.imshow(np.array(segmented_image[i], dtype=int))
	plt.show()

	plt.imshow(mask_image)
	plt.show()
	"""
	return mask_image


def cell_segmentation(image_shg, image_pl, image_tran, scale=1.5, sigma=0.8, alpha=1.0,
			min_size=750, edges=False):

	"Return binary filter for cellular identification"
	
	image_scale_shg = rescale(image_shg, scale)
	image_scale_pl = rescale(image_pl, scale)
	image_scale_tran = rescale(image_tran, scale)

	image_stack = np.stack((image_scale_shg, image_scale_pl, image_scale_tran), axis=-1)
	mask_image = BD_filter(image_stack)
	mask_image = resize(mask_image, image_shg.shape)

	cells = []
	areas = []
	cell_labels = measure.label(mask_image)

	for cell in measure.regionprops(cell_labels, intensity_image=image_pl):
		cell_check = True

		if edges:
			edge_check = (cell.bbox[0] != 0) * (cell.bbox[1] != 0)
			edge_check *= (cell.bbox[2] != mask_image.shape[0])
			edge_check *= (cell.bbox[3] != mask_image.shape[1])

			cell_check *= edge_check

		cell_check *= cell.area >= min_size

		if cell_check:
			cells.append(cell)
			areas.append(cell.area)

	indices = np.argsort(areas)[::-1]
	sorted_cells = [cells[i] for i in indices]

	return sorted_cells



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


def network_extraction(image_shg, network_name='network', scale=1.25, sigma=0.5, p_denoise=(2, 25), 
			threads=8):
	"""
	Extract fibre network using modified FIRE algorithm
	"""

	print("Performing NL Denoise using local windows {} {}".format(*p_denoise))
	image_nl = nl_means(image_shg, p_denoise=p_denoise)

	"Call FIRE algorithm to extract full image network"
	print("Calling FIRE algorithm using image scale {}".format(scale))
	Aij = FIRE(image_nl, scale=scale, sigma=sigma, max_threads=threads)
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


def fibre_segmentation(image_shg, networks, networks_red, area_threshold=200, iterations=6):

	n_net = len(networks)
	fibres = []

	iterator = zip(np.arange(n_net), networks, networks_red)

	"Segment image based on connected components in network"
	for i, network, network_red in iterator:

		label_image = np.zeros(image_shg.shape, dtype=int)
		label_image = draw_network(network, label_image, 1)

		dilated_image = binary_dilation(label_image, iterations=iterations)
		filled_image = remove_small_holes(dilated_image, area_threshold=area_threshold)
		smoothed_image = gaussian_filter(filled_image, sigma=0.5)
		binary_image = np.where(smoothed_image, 1, 0)

		segment = measure.regionprops(binary_image, intensity_image=image_shg)[0]
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
