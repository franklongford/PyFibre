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

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_closing, binary_opening

from skimage import measure, draw
from skimage.transform import rescale, resize
from skimage.feature import greycomatrix, greycoprops
from skimage.morphology import remove_small_objects, remove_small_holes

import utilities as ut
from filters import tubeness, hysteresis
from extraction import FIRE, waviness_analysis, simplify_network
from analysis import fourier_transform_analysis, tensor_analysis, angle_analysis
from preprocessing import nl_means


def create_binary_image(segments, shape):

	binary_image = np.zeros(shape)

	for segment in segments:
		minr, minc, maxr, maxc = segment.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]
		binary_image[(indices[0], indices[1])] += segment.image

	binary_image = np.where(binary_image, 1, 0)

	return binary_image


def find_holes(image, sigma=0.8, alpha=1.0, min_size=1250, iterations=1):

	image_TB = tubeness(image, sigma=sigma)
	
	image_hyst = hysteresis(image_TB, alpha=alpha)
	image_hyst = remove_small_objects(image_hyst)
	image_hyst = binary_closing(image_hyst, iterations=iterations)

	image_hole = remove_small_objects(~image_hyst, min_size=min_size)
	image_hole = binary_opening(image_hole, iterations=iterations)
	image_hole = binary_fill_holes(image_hole)
	
	return image_hole


def hole_extraction(image_shg, image_pl, fibres, scale=2, sigma=0.8, alpha=1.0, min_size=1250, edges=False):

	image_fibre = create_binary_image(fibres, image_pl.shape)
	image_hole_fibre = np.where(image_fibre, 0, 1)

	image_shg_big = rescale(image_shg, scale)	
	image_hole_shg = find_holes(image_shg_big, sigma=sigma, alpha=0.9, 
					min_size=min_size, iterations=1) 
	image_hole_shg = resize(image_hole_shg, image_shg.shape)
	
	image_back_big = rescale(np.sqrt(image_shg * image_pl), scale)	
	background = find_holes(image_back_big, sigma=sigma, alpha=0.4, 
					min_size=min_size, iterations=1) 
	background = resize(background, image_pl.shape)

	image_diff = np.where((image_hole_fibre * image_hole_shg) != background, 1, 0)

	holes = []
	hole_labels = measure.label(image_diff)

	for hole in measure.regionprops(hole_labels, intensity_image=image_pl):
		hole_check = True

		if edges:
			edge_check = (hole.bbox[0] != 0) * (hole.bbox[1] != 0)
			edge_check *= (hole.bbox[2] != image_hole.shape[0])
			edge_check *= (hole.bbox[3] != image_hole.shape[1])

			hole_check *= edge_check

		hole_check *= hole.area >= min_size

		if hole_check: holes.append(hole)

	return holes


def hole_analysis(image, holes):

	l_holes = len(holes)

	hole_areas = np.zeros(l_holes)
	hole_hu = np.zeros((l_holes, 7))
	hole_mean = np.zeros(l_holes)
	hole_std = np.zeros(l_holes)
	hole_entropy = np.zeros(l_holes)

	hole_glcm_contrast = np.zeros(l_holes)
	hole_glcm_dissim = np.zeros(l_holes)
	hole_glcm_corr = np.zeros(l_holes)
	hole_glcm_homo = np.zeros(l_holes)
	hole_glcm_energy = np.zeros(l_holes)
	hole_glcm_IDM = np.zeros(l_holes)
	hole_glcm_variance = np.zeros(l_holes)
	hole_glcm_cluster = np.zeros(l_holes)
	hole_glcm_entropy = np.zeros(l_holes)

	hole_linear = np.zeros(l_holes)
	hole_eccent = np.zeros(l_holes)

	for i, hole in enumerate(holes):
		
		minr, minc, maxr, maxc = hole.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		hole_image = image[(indices[0], indices[1])] * hole.image

		hole_areas[i] = hole.area
		hole_hu[i] = hole.moments_hu

		glcm = greycomatrix((hole_image * 255.999).astype('uint8'),
		                 [1, 2], [0, np.pi/4, np.pi/2, np.pi*3/4], 256,
		                 symmetric=True, normed=True)

		hole_mean[i] = np.mean(hole_image)
		hole_std[i] = np.std(hole_image)
		hole_entropy[i] = measure.shannon_entropy(hole_image)

		hole_glcm_contrast[i] = greycoprops_edit(glcm, 'contrast').mean()
		hole_glcm_homo[i] = greycoprops_edit(glcm, 'homogeneity').mean()
		hole_glcm_dissim[i] = greycoprops_edit(glcm, 'dissimilarity').mean()
		hole_glcm_corr[i] = greycoprops_edit(glcm, 'correlation').mean()
		hole_glcm_energy[i] = greycoprops_edit(glcm, 'energy').mean()
		hole_glcm_IDM[i] = greycoprops_edit(glcm, 'IDM').mean()
		hole_glcm_variance[i] = greycoprops_edit(glcm, 'variance').mean()
		hole_glcm_cluster[i] = greycoprops_edit(glcm, 'cluster').mean()
		hole_glcm_entropy[i] = greycoprops_edit(glcm, 'entropy').mean()

		hole_linear[i] = 1 - hole.equivalent_diameter / hole.perimeter
		hole_eccent[i] = hole.eccentricity

	return (hole_areas, hole_mean, hole_std, hole_entropy, hole_glcm_contrast,
		hole_glcm_homo, hole_glcm_dissim, hole_glcm_corr, hole_glcm_energy,
		hole_glcm_IDM, hole_glcm_variance, hole_glcm_cluster, hole_glcm_entropy,
		hole_linear, hole_eccent, hole_hu) 


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


def segment_analysis(image_shg, image_pl, segment, n_tensor, anis_map, angle_map):

	minr, minc, maxr, maxc = segment.bbox
	indices = np.mgrid[minr:maxr, minc:maxc]

	segment_image_shg = image_shg[(indices[0], indices[1])]
	segment_image_pl = image_pl[(indices[0], indices[1])]
	#segment_image_comb = np.sqrt(segment_image_shg * segment_image_pl)
	segment_anis_map = anis_map[(indices[0], indices[1])]
	segment_angle_map = angle_map[(indices[0], indices[1])]
	segment_n_tensor = n_tensor[(indices[0], indices[1])]

	_, _, segment_fourier_sdi = fourier_transform_analysis(segment_image_shg)
	segment_angle_sdi = angle_analysis(segment_angle_map, segment_anis_map)

	segment_mean = np.mean(segment_image_shg)
	segment_std = np.std(segment_image_shg)
	segment_entropy = measure.shannon_entropy(segment_image_shg)

	segment_anis, _ , _ = tensor_analysis(np.mean(segment_n_tensor, axis=(0, 1)))
	segment_anis = segment_anis[0]
	segment_pix_anis = np.mean(segment_anis_map)

	segment_area = np.sum(segment.image)
	segment_linear = 1 - segment.equivalent_diameter / segment.perimeter
	segment_eccent = segment.eccentricity
	segment_density = np.sum(segment_image_shg * segment.image) / segment_area
	segment_coverage = np.mean(segment.image)
	segment_hu = segment.moments_hu

	glcm = greycomatrix((segment_image_shg * segment.image * 255.999).astype('uint8'),
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
	networks = []
	networks_red = []
	for i, component in enumerate(nx.connected_components(Aij)):
		subgraph = Aij.subgraph(component)

		networks.append(subgraph)
		networks_red.append(simplify_network(subgraph))

	return networks, networks_red


def fibre_extraction(image_shg, networks, networks_red, area_threshold=200):

	n_net = len(networks)
	fibres = []

	iterator = zip(np.arange(n_net), networks, networks_red)

	"Segment image based on connected components in network"
	for i, network, network_red in iterator:

		label_image = np.zeros(image_shg.shape, dtype=int)
		label_image = draw_network(network, label_image, 1)
		dilated_image = binary_dilation(label_image, iterations=8)
		filled_image = remove_small_holes(dilated_image, area_threshold=area_threshold)
		smoothed_image = gaussian_filter(filled_image, sigma=0.25)
		binary_image = np.where(smoothed_image, 1, 0)

		segment = measure.regionprops(binary_image, intensity_image=image_shg)[0]
		area = np.sum(segment.image)

		minr, minc, maxr, maxc = segment.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		#print(network.number_of_nodes(), area, 1E-2 * image_shg.size)

		segment.label = (i + 1)
		fibres.append(segment)
	
	"""	
	"Sort segments ranked by area"
	indices = np.argsort(areas)
	sorted_areas = areas[indices]
	sorted_segments = [segments[i] for i in indices]
	sorted_networks = [networks[i] for i in indices]
	sorted_networks_red = [networks_red[i] for i in indices]
	"""

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

	from extraction import updated_waviness_analysis

	network_waviness, network_waviness_std = updated_waviness_analysis(network)

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

	return (network_waviness, network_waviness_std, network_degree, 
			network_eigen, network_connect)


def total_analysis(image_shg, image_pl, networks, networks_red, 
				segments, n_tensor, anis_map, angle_map):
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
	
	network_waviness = np.empty(l_regions)
	network_waviness_std = np.empty(l_regions)
	network_degree = np.empty(l_regions)
	network_eigen = np.empty(l_regions)
	network_connect = np.empty(l_regions)
	#network_loc_eff = np.empty(l_regions)
	#network_cluster = np.empty(l_regions)

	iterator = zip(np.arange(l_regions), networks, networks_red, segments)

	for i, network, network_red, segment in iterator:

		#if segment.filled_area >= 1E-2 * image_shg.size:

		metrics = segment_analysis(image_shg, image_pl, segment, n_tensor, anis_map,
						angle_map)

		(segment_fourier_sdi[i], segment_angle_sdi[i], segment_anis[i], 
		segment_pix_anis[i], segment_area[i], segment_linear[i], segment_eccent[i], 
		segment_density[i], segment_coverage[i], segment_mean[i], segment_std[i],
		segment_entropy[i], segment_glcm_contrast[i], segment_glcm_homo[i], segment_glcm_dissim[i], 
		segment_glcm_corr[i], segment_glcm_energy[i], segment_glcm_IDM[i], 
		segment_glcm_variance[i], segment_glcm_cluster[i], segment_glcm_entropy[i],
		segment_hu[i]) = metrics

		metrics = network_analysis(network, network_red)

		(network_waviness[i], network_waviness_std[i], network_degree[i], network_eigen[i],
			network_connect[i]) = metrics


	return (segment_fourier_sdi, segment_angle_sdi, segment_anis, segment_pix_anis, 
		segment_area, segment_linear, segment_eccent, segment_density, segment_coverage,
		segment_mean, segment_std, segment_entropy, segment_glcm_contrast, 
		segment_glcm_homo, segment_glcm_dissim, segment_glcm_corr, segment_glcm_energy, 
		segment_glcm_IDM, segment_glcm_variance, segment_glcm_cluster, segment_glcm_entropy,
		segment_hu, network_waviness, network_waviness_std, network_degree, network_eigen, 
		network_connect)
