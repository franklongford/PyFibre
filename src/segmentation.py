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
from networkx.algorithms import approximation as approx
from networkx.algorithms.efficiency import local_efficiency, global_efficiency

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_closing, binary_opening

from skimage import measure, draw
from skimage.feature import greycomatrix, greycoprops
from skimage.morphology import remove_small_objects

import utilities as ut
from filters import tubeness, hysteresis
from extraction import FIRE, adj_analysis
from analysis import fourier_transform_analysis, tensor_analysis
from preprocessing import nl_means


def hole_extraction(image, sigma=0.75, alpha=0.7, min_size=800):

	image_TB = tubeness(image, sigma=sigma)
	
	image_hyst = hysteresis(image_TB, alpha=alpha)
	image_hyst = remove_small_objects(image_hyst)
	image_hyst = binary_closing(image_hyst)

	image_hole = remove_small_objects(~image_hyst, min_size=min_size)
	image_hole = binary_opening(image_hole)
	image_hole = binary_fill_holes(image_hole)

	holes = []
	hole_labels = measure.label(image_hole)

	for hole in measure.regionprops(hole_labels, intensity_image=image):
		edge_check = (hole.bbox[0] != 0) * (hole.bbox[1] != 0)
		edge_check *= (hole.bbox[2] != image_hole.shape[0])
		edge_check *= (hole.bbox[3] != image_hole.shape[1])

		#if edge_check: holes.append(hole)
		holes.append(hole)

	return holes, hole_labels


def hole_analysis(image, holes):

	l_holes = len(holes)

	hole_areas = np.zeros(l_holes)
	hole_hu = np.zeros((l_holes, 7))
	hole_contrast = np.zeros(l_holes)
	hole_dissim = np.zeros(l_holes)
	hole_corr = np.zeros(l_holes)
	hole_homo = np.zeros(l_holes)
	hole_energy = np.zeros(l_holes)

	for i, hole in enumerate(holes):
		
		minr, minc, maxr, maxc = hole.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		hole_image = image[(indices[0], indices[1])]

		hole_areas[i] = hole.area
		hole_hu[i] = hole.moments_hu

		glcm = greycomatrix((hole_image * 255.999).astype('uint8'),
		                 [1, 2], [0, np.pi/4, np.pi/2, np.pi*3/4], 256,
		                 symmetric=True, normed=True)

		hole_contrast[i] = greycoprops(glcm, 'contrast').mean()
		hole_homo[i] = greycoprops(glcm, 'homogeneity').mean()
		hole_dissim[i] = greycoprops(glcm, 'dissimilarity').mean()
		hole_corr[i] = greycoprops(glcm, 'correlation').mean()
		hole_energy[i] = greycoprops(glcm, 'energy').mean()

	return hole_areas, hole_contrast, hole_homo, hole_dissim, hole_corr, hole_energy, hole_hu


def segment_analysis(image_shg, image_pl, segment, n_tensor, anis_map):

	minr, minc, maxr, maxc = segment.bbox
	indices = np.mgrid[minr:maxr, minc:maxc]

	segment_image_shg = image_shg[(indices[0], indices[1])]
	segment_image_pl = image_pl[(indices[0], indices[1])]
	#segment_image_comb = np.sqrt(segment_image_shg * segment_image_pl)
	segment_anis_map = anis_map[(indices[0], indices[1])]
	segment_n_tensor = n_tensor[(indices[0], indices[1])]

	_, _, segment_sdi = fourier_transform_analysis(segment_image_shg)
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

	glcm = greycomatrix((segment_image_shg * 255.999).astype('uint8'),
                         [1, 2], [0, np.pi/4, np.pi/2, np.pi*3/4], 256,
                         symmetric=True, normed=True)

	segment_contrast = greycoprops(glcm, 'contrast').mean()
	segment_homo = greycoprops(glcm, 'homogeneity').mean()
	segment_dissim = greycoprops(glcm, 'dissimilarity').mean()
	segment_corr = greycoprops(glcm, 'correlation').mean()
	segment_energy = greycoprops(glcm, 'energy').mean()

	return (segment_sdi, segment_entropy, segment_anis, segment_pix_anis, 
		segment_area, segment_linear, segment_eccent, segment_density, 
		segment_coverage, segment_contrast, segment_homo, segment_dissim, 
		segment_corr, segment_energy, segment_hu)


def network_extraction(image, network_name='network', sigma=0.5, p_denoise=(5, 30), 
			ow_network=False, threads=8):
	"""
	Extract fibre network using modified FIRE algorithm
	"""

	"Try loading saved graph opbject"
	try: Aij = nx.read_gpickle(network_name + "_network.pkl")
	except IOError: ow_network = True

	"Else, use modified FIRE algorithm to extract network"
	if ow_network:
		print("Performing NL Denoise using local windows {} {}".format(*p_denoise))
		image_TB = tubeness(image, 2 * sigma)
		image_nl = nl_means(image_TB, p_denoise=p_denoise)
		"Apply tubeness transform to enhance image fibres"
		#image_TB = tubeness(image_nl, sigma)
		"Call FIRE algorithm to extract full image network"
		Aij = FIRE(image_nl, sigma=sigma, max_threads=threads)
		nx.write_gpickle(Aij, network_name + "_network.pkl")

	segmented_image = np.zeros(image.shape, dtype=int)
	areas = np.empty((0,), dtype=float)
	networks = []
	segments = []

	"Segment image based on connected components in network"
	for i, component in enumerate(nx.connected_components(Aij)):
		subgraph = Aij.subgraph(component)
		if subgraph.number_of_nodes() > 3:

			label_image = np.zeros(image.shape, dtype=int)
			label_image = draw_network(subgraph, label_image, 1)
			dilated_image = binary_dilation(label_image, iterations=6)
			filled_image = np.array(binary_fill_holes(dilated_image),
					    dtype=int)

			segment = measure.regionprops(filled_image)[0]
			area = np.sum(segment.image)

			minr, minc, maxr, maxc = segment.bbox
			indices = np.mgrid[minr:maxr, minc:maxc]

			if area >= 1E-2 * image.size:
				segmented_image[(indices[0], indices[1])] = segment.image
				segment.label = (i + 1)
				areas = np.concatenate((areas, [area]))
				segments.append(segment)
				networks.append(subgraph)
		
	"Sort segments ranked by area"
	indices = np.argsort(areas)
	sorted_areas = areas[indices]
	sorted_segments = [segments[i] for i in indices]
	sorted_networks = [networks[i] for i in indices]

	return segmented_image, sorted_networks, sorted_segments


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


def network_analysis(image_shg, image_pl, networks, segments, n_tensor, anis_map):
	"""
	Analyse extracted fibre network
	"""
	l_regions = len(segments)

	segment_sdi = np.zeros(l_regions)
	segment_entropy = np.zeros(l_regions)
	segment_anis = np.zeros(l_regions)
	segment_pix_anis = np.zeros(l_regions)

	segment_area = np.zeros(l_regions)
	segment_linear = np.zeros(l_regions)
	segment_eccent = np.zeros(l_regions)
	segment_density = np.zeros(l_regions)
	segment_coverage = np.zeros(l_regions)

	segment_contrast = np.zeros(l_regions)
	segment_dissim = np.zeros(l_regions)
	segment_corr = np.zeros(l_regions)
	segment_homo = np.zeros(l_regions)
	segment_energy = np.zeros(l_regions)
	segment_hu = np.zeros((l_regions, 7))
	
	network_waviness = np.empty(l_regions)
	network_degree = np.empty(l_regions)
	network_eigen = np.empty(l_regions)
	network_connect = np.empty(l_regions)
	network_loc_eff = np.empty(l_regions)
	network_cluster = np.empty(l_regions)

	waviness_time = 0
	degree_time = 0
	central_time = 0
	connect_time = 0
	loc_eff_time = 0
	cluster_time = 0

	iterator = zip(np.arange(l_regions), networks, segments)

	for i, network, segment in iterator:

		metrics = segment_analysis(image_shg, image_pl, segment, n_tensor, anis_map)

		(segment_sdi[i], segment_entropy[i], segment_anis[i], segment_pix_anis[i], 
		segment_area[i], segment_linear[i], segment_eccent[i], segment_density[i], 
		segment_coverage[i], segment_contrast[i], segment_homo[i], segment_dissim[i], 
		segment_corr[i], segment_energy[i], segment_hu[i]) = metrics

		start = time.time()
		network_waviness[i] = adj_analysis(network)
		stop1 = time.time()
		waviness_time += stop1-start

		try: network_degree[i] = nx.degree_pearson_correlation_coefficient(network)**2
		except: network_degree[i] = None
		stop2 = time.time()
		degree_time += stop2-stop1

		try: network_eigen[i] = np.real(nx.adjacency_spectrum(network).max())
		except: network_eigen[i] = None
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

		try: network_cluster[i] = approx.clustering_coefficient.average_clustering(network)
		except: network_cluster[i] = None
		stop6 = time.time()
		cluster_time += stop6-stop5

	#print('Network Waviness = {} s'.format(waviness_time))
	#print('Network Degree = {} s'.format(degree_time))
	#print('Network Eigenvalue = {} s'.format(central_time))
	#print('Network Conectivity = {} s'.format(connect_time))
	#print('Network Local Efficiency = {} s'.format(loc_eff_time))
	#print('Network Clustering = {} s'.format(cluster_time))

	return (segment_sdi, segment_entropy, segment_anis, segment_pix_anis, 
		segment_area, segment_linear, segment_eccent, segment_density, segment_coverage,
		segment_contrast, segment_homo, segment_dissim, segment_corr, segment_energy, segment_hu,
		network_waviness, network_degree, network_eigen, network_connect,
		network_loc_eff, network_cluster)
