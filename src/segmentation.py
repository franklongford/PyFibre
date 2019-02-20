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
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

from skimage import measure, draw
from skimage.feature import greycomatrix, greycoprops

import utilities as ut
from filters import tubeness
from extraction import FIRE, adj_analysis
from analysis import fourier_transform_analysis, tensor_analysis
from preprocessing import nl_means


def network_extraction(image, network_name='network', sigma=1.0, p_denoise=(5, 30), 
						ow_network=False, threads=8):
	"""
	Extract fibre network using modified FIRE algorithm
	"""

	"Try loading saved graph opbject"
	try: Aij = nx.read_gpickle(network_name + "_network.pkl")
	except IOError: ow_network = True

	"Else, use modified FIRE algorithm to extract network"
	if ow_network:
		"Apply tubeness transform to enhance image fibres"
		image_nl = nl_means(image, p_denoise=p_denoise)
		image_TB = tubeness(image_nl, sigma)
		"Call FIRE algorithm to extract full image network"
		Aij = FIRE(image_TB, sigma=sigma, max_threads=threads)
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

			if area >= 2E-3 * image.size:
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

	return segmented_image, sorted_networks, sorted_areas, sorted_segments


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


def network_analysis(image, rgb_image, networks, segments, n_tensor, anis_map):
	"""
	Analyse extracted fibre network
	"""
	l_regions = len(segments)

	segment_sdi = np.zeros(l_regions)
	segment_entropy = np.zeros(l_regions)
	segment_anis = np.zeros(l_regions)
	segment_pix_anis = np.zeros(l_regions)

	segment_linear = np.zeros(l_regions)
	segment_eccent = np.zeros(l_regions)
	segment_density = np.zeros(l_regions)
	segment_coverage = np.zeros(l_regions)

	segment_contrast = np.zeros(l_regions)
	segment_dissim = np.zeros(l_regions)
	segment_corr = np.zeros(l_regions)
	segment_homo = np.zeros(l_regions)
	segment_energy = np.zeros(l_regions)
	
	network_waviness = np.empty(l_regions)
	network_degree = np.empty(l_regions)
	network_central = np.empty(l_regions)
	network_connect = np.empty(l_regions)
	network_loc_eff = np.empty(l_regions)
	network_glob_eff = np.empty(l_regions)

	waviness_time = 0
	degree_time = 0
	central_time = 0
	connect_time = 0
	loc_eff_time = 0
	glob_eff_time = 0

	iterator = zip(np.arange(l_regions), networks, segments)

	for i, network, segment in iterator:

		minr, minc, maxr, maxc = segment.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		segment_image = image[(indices[0], indices[1])]
		segment_rgb_image = rgb_image[(indices[0], indices[1])]
		segment_anis_map = anis_map[(indices[0], indices[1])]
		segment_n_tensor = n_tensor[(indices[0], indices[1])]

		_, _, sdi = fourier_transform_analysis(segment_image)
		segment_sdi[i] = sdi
		segment_entropy[i] = measure.shannon_entropy(segment_image)

		segment_anis[i], _ , _ = tensor_analysis(np.mean(segment_n_tensor, axis=(0, 1)))
		segment_pix_anis[i] = np.mean(segment_anis_map)

		segment_linear[i] += 1 - segment.equivalent_diameter / segment.perimeter
		segment_eccent[i] += segment.eccentricity
		segment_density[i] += np.sum(segment_image * segment.image) / np.sum(segment.image)
		segment_coverage[i] += np.mean(segment.image)

		glcm = greycomatrix((segment_image * 255.999).astype('uint8'),
                             [1, 2], [0, np.pi/4, np.pi/2, np.pi*3/4], 256, symmetric=True, normed=True)

		segment_contrast[i] += greycoprops(glcm, 'contrast').mean()
		segment_homo[i] += greycoprops(glcm, 'homogeneity').mean()
		segment_dissim[i] += greycoprops(glcm, 'dissimilarity').mean()
		segment_corr[i] += greycoprops(glcm, 'correlation').mean()
		segment_energy[i] += greycoprops(glcm, 'energy').mean()

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

	return (segment_sdi, segment_entropy, segment_anis, segment_pix_anis, 
		segment_linear, segment_eccent, segment_density, segment_coverage,
		segment_contrast, segment_homo, segment_dissim, segment_corr, segment_energy,
		network_waviness, network_degree, network_central, 
		network_connect, network_loc_eff)
