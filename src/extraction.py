"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import numpy as np
import sys
import copy
import time
import itertools
import threading

from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import coo_matrix, csr_matrix, csgraph

from skimage.feature import (structure_tensor, hessian_matrix, hessian_matrix_eigvals)
from skimage.morphology import disk, square, local_maxima, binary_erosion, remove_small_objects
from skimage.transform import rescale
from skimage.filters import rank

import networkx as nx

from utilities import ring, numpy_remove, clear_border
from filters import tubeness, hysteresis


def check_2D_arrays(array1, array2, thresh=1):
	"return indices where values of array1 are within thresh distance of array2"

	array1_mat = np.tile(array1, (1, array2.shape[0]))\
	                .reshape(array1.shape[0], array2.shape[0], 2)    
	array2_mat = np.tile(array2, (array1.shape[0], 1))\
	                .reshape(array1.shape[0], array2.shape[0], 2)

	diff = np.sum((array1_mat - array2_mat)**2, axis=2)
	array1_indices = np.argwhere(diff <= thresh**2)[:,0]
	array2_indices = np.argwhere(diff <= thresh**2)[:,1]

	return array1_indices, array2_indices


def distance_matrix(node_coord):
	"calculate distances between each index value of node_coord"

	node_coord_matrix = np.tile(node_coord, (node_coord.shape[0], 1))\
	                    .reshape(node_coord.shape[0], node_coord.shape[0], node_coord.shape[1])
	d_node = node_coord_matrix - np.transpose(node_coord_matrix, (1, 0, 2))
	r2_node = np.sum(d_node**2, axis=2)

	return d_node, r2_node


def reduce_coord(coord, values, thresh=1):
	"""
	Find elements in coord that lie within thresh distance of eachother. Remove
	element with lowest corresponding value. 
	"""

	if coord.shape[0] <= 1: return coord

	thresh = np.sqrt(2 * thresh**2)
	r_coord = cdist(coord, coord)

	del_coord = np.argwhere((r_coord <= thresh) - np.identity(coord.shape[0]))
	del_coord = del_coord[np.arange(0, del_coord.shape[0], 2)]
	indices = np.stack((values[del_coord[:,0]], 
			    values[del_coord[:,1]])).argmax(axis=0)
	del_coord = [a[i] for a, i in zip(del_coord, indices)]

	coord = np.delete(coord, del_coord, 0)

	return coord


def cos_sin_theta_2D(vector, r_vector):
	"""
	cos_sin_theta_2D(vector, r_vector)

	Returns cosine and sine of angles of intersecting vectors betwen even and odd indicies

	Parameters
	----------

	vector:  array_like, (float); shape=(n_vector, n_dim)
	    Array of displacement vectors between connecting beads

	r_vector: array_like, (float); shape=(n_vector)
	    Array of radial distances between connecting beads

	Returns
	-------

	cos_the:  array_like (float); shape=(n_vector/2)
	    Cosine of the angle between each pair of displacement vectors

	sin_the: array_like (float); shape=(n_vector/2)
	    Sine of the angle between each pair of displacement vectors

	r_prod: array_like (float); shape=(n_vector/2)
	    Product of radial distance between each pair of displacement vectors
	"""

	n_vector = int(vector.shape[0])
	n_dim = vector.shape[1]

	temp_vector = np.reshape(vector, (int(n_vector/2), 2, n_dim))

	"Calculate |rij||rjk| product for each pair of vectors"
	r_prod = np.prod(np.reshape(r_vector, (int(n_vector/2), 2)), axis = 1)

	"Form dot product of each vector pair rij*rjk in vector array corresponding to an angle"
	dot_prod = np.sum(np.prod(temp_vector, axis=1), axis=1)

	"Calculate cos(theta) for each angle"
	cos_the = dot_prod / r_prod

	return cos_the


def new_branches(image, coord, ring_filter, max_thresh=0.2):

        filtered = image * ring_filter
        branch_coord = np.argwhere(local_maxima(filtered) * image >= max_thresh)
        branch_coord = reduce_coord(branch_coord, 
				    image[branch_coord[:,0], branch_coord[:,1]])
        
        n_branch = branch_coord.shape[0]
        branch_vector = np.tile(coord, (n_branch, 1)) - branch_coord
        branch_r = np.sqrt(np.sum(branch_vector**2, axis=1))
        
        return branch_coord, branch_vector, branch_r


def branch_angles(direction, branch_vector, branch_r):

	n_branch = branch_vector.shape[0]
	dir_vector = np.tile(direction, (n_branch, 1))
	dir_r = np.ones(n_branch)

	combined_vector = np.hstack((branch_vector, dir_vector)).reshape(n_branch*2, 2)
	combined_r = np.column_stack((branch_r, dir_r)).flatten()
	cos_the = cos_sin_theta_2D(combined_vector, combined_r)

	return cos_the


def transfer_edges(Aij, source, target):

	for node in list(Aij.adj[source]):
		Aij.remove_edge(node, source)
		if node != source:
			Aij.add_edge(node, target)
			Aij[node][target]['r'] = np.sqrt(((Aij.nodes[target]['xy'] - Aij.nodes[node]['xy'])**2).sum())


class Fibre(nx.Graph):

	def __init__(self, nodes, edges=[], direction=[0,0], growing=True,
			fibre_l=0, euclid_l=0):

		super().__init__()
		self.add_nodes_from(nodes)
		self.add_edges_from(edges)
		self.direction = direction
		self.growing = growing
		self.fibre_l = fibre_l
		self.euclid_l = euclid_l

		self.node_list = list(self.nodes)

    
def grow(fibre, image, Aij, tot_node_coord, lmp_thresh, theta_thresh, r_thresh):
	"""
	Grow fibre object along network

	Parameters
	----------
	
	fibre: nx.Graph
	    Object of class Fibre to be grown in network
	image:  array_like, (float); shape=(nx, ny)
	    Image to perform FIRE upon
	Aij: nx.Graph
	    Networkx graph object representing fibre network
	tot_node_coord: array_like
	    Array of full coordinates (x, y) of nodes in graph Aij
	lmp_thresh: float
	    Minimum distance pixel threshold to be classed as lmp point
	theta_thresh: float
	    Maximum radian deviation of new lmp from fibre trajectory
	r_thresh: float
	    Maximum length of edges between nodes
	"""

	nuc = Aij.nodes[fibre]['nuc']
	connect = list(Aij[fibre].keys())[0]

	start_coord = Aij.nodes[nuc]['xy']
	end_coord = Aij.nodes[fibre]['xy']

	ring_filter = ring(np.zeros(image.shape), end_coord, np.arange(2, 3), 1)
	branch_coord, branch_vector, branch_r = new_branches(image, end_coord, 
			                                 ring_filter, lmp_thresh)
	cos_the = branch_angles(Aij.nodes[fibre]['direction'], branch_vector, branch_r)
	indices = np.argwhere(abs(cos_the + 1) <= theta_thresh)

	if indices.size == 0: 
		Aij.nodes[fibre]['growing'] = False

		if Aij[fibre][connect]['r'] <= r_thresh / 10: transfer_edges(Aij, fibre, connect)
		
		return

	branch_coord = branch_coord[indices]
	branch_vector = branch_vector[indices]
	branch_r = branch_r[indices]

	close_nodes, _ = check_2D_arrays(tot_node_coord, branch_coord, 1)
	close_nodes = numpy_remove(close_nodes, list(Aij[fibre].keys()))

	if close_nodes.size != 0:

		new_end = close_nodes.min()

		end_coord = Aij.nodes[connect]['xy']
		new_end_coord = Aij.nodes[new_end]['xy']

		transfer_edges(Aij, fibre, new_end)

		Aij.nodes[fibre]['growing'] = False
	
	else:
		index = branch_r.argmax()

		new_end_coord = branch_coord[index].flatten()
		new_end_vector = new_end_coord - Aij.nodes[connect]['xy']
		new_end_r = np.sqrt((new_end_vector**2).sum())

		new_dir_vector = new_end_coord - start_coord
		new_dir_r = np.sqrt((new_dir_vector**2).sum())

		if new_end_r >= r_thresh:

			new_end = Aij.number_of_nodes()

			Aij.add_node(new_end)
			Aij.add_edge(fibre, new_end)
		
			Aij.nodes[new_end]['xy'] = new_end_coord
			Aij.nodes[new_end]['nuc'] = nuc

			Aij[fibre][new_end]['r'] = np.sqrt(((new_end_coord - end_coord)**2).sum())
			Aij.nodes[new_end]['direction'] = (new_dir_vector / new_dir_r)

			Aij.nodes[fibre]['growing'] = False
			Aij.nodes[new_end]['growing'] = True 

		else: 
			Aij.nodes[fibre]['xy'] = new_end_coord
			Aij[fibre][connect]['r'] = new_end_r
			Aij.nodes[fibre]['direction'] = (new_dir_vector / new_dir_r)


def FIRE(image, scale=1, alpha=0.70, sigma=0.5, nuc_thresh=2, nuc_rad=11, lmp_thresh=0.15, 
             angle_thresh=70, r_thresh=8, max_threads=8):
	"""
	FIRE algorithm to extract fibre network

	Parameters
	----------

	image:  array_like, (float); shape=(nx, ny)
	    Image to perform FIRE upon
	sigma: float
	    Gaussian standard deviation to filter distance image
	nuc_thresh: float
	    Minimum distance pixel threshold to be classed as nucleation point
	nuc_rad: float
	    Minimum pixel radii between nucleation points
	lmp_thresh: float
	    Minimum distance pixel threshold to be classed as lmp point
	angle_thresh: float
	    Maximum angular deviation of new lmp from fibre trajectory
	r_thresh: float
	    Maximum length of edges between nodes
	max_threads: ont
	    Maximum number of threads for multithreading routine

	Returns
	-------

	Aij: nx.Graph
	    Networkx graph object representing fibre network

	"""

	"Prepare input image to gain distance matrix of foreground from background"

	image_scale = rescale(image, scale)

	sigma *= scale

	"Apply tubeness transform to enhance image fibres"
	image_TB = tubeness(image_scale, 2 * sigma)
	threshold = hysteresis(image_TB, alpha=alpha)
	cleaned = remove_small_objects(threshold)
	distance = distance_transform_edt(cleaned)
	smoothed = gaussian_filter(distance, sigma=sigma)
	smoothed = clear_border(smoothed)

	"Set distance and angle thresholds for fibre iterator"
	nuc_thresh = np.min([nuc_thresh, 1E-1 * scale**2 * smoothed.max()])
	lmp_thresh = np.min([lmp_thresh, 1E-2 * scale**2 * smoothed[np.nonzero(smoothed)].mean()])
	theta_thresh = np.cos((180-angle_thresh) * np.pi / 180) + 1
	r_thresh = int(r_thresh * scale)
	nuc_rad = int(nuc_rad * scale)

	print("Maximum distance = {}".format(smoothed.max()))
	print("Mean distance = {}".format(smoothed[np.nonzero(smoothed)].mean()))
	print("Using thresholds:\n nuc = {} pix\n lmp = {} pix\n angle = {} deg\n edge = {} pix".format(
		    nuc_thresh, lmp_thresh, angle_thresh, r_thresh))

	"Get global maxima for smoothed distance matrix"
	maxima = local_maxima(smoothed, connectivity=nuc_rad)
	nuc_node_coord = reduce_coord(np.argwhere(maxima * smoothed >= nuc_thresh),
		            smoothed[np.where(maxima * smoothed >= nuc_thresh)], r_thresh)

	"Set up network arrays"
	n_nuc = nuc_node_coord.shape[0]

	print("No. nucleation nodes = {}".format(n_nuc))

	Aij = nx.Graph()
	Aij.add_nodes_from(np.arange(n_nuc))

	"Iterate through nucleation points"
	index_m = n_nuc
	for nuc, nuc_coord in enumerate(nuc_node_coord):

		Aij.nodes[nuc]['xy'] = nuc_coord
		Aij.nodes[nuc]['nuc'] = nuc
		Aij.nodes[nuc]['growing'] = False

		ring_filter = ring(np.zeros(smoothed.shape), nuc_coord, [r_thresh // 2], 1)
		lmp_coord, lmp_vectors, lmp_r = new_branches(smoothed, nuc_coord, ring_filter, 
				                             lmp_thresh)
		n_lmp = lmp_coord.shape[0]

		Aij.add_nodes_from(index_m + np.arange(n_lmp))
		Aij.add_edges_from([*zip(nuc * np.ones(n_lmp, dtype=int), 
				         index_m + np.arange(n_lmp))])

		iterator = zip(lmp_coord, lmp_vectors, lmp_r, index_m + np.arange(n_lmp))

		for xy, vec, r, lmp in iterator:

			Aij.nodes[lmp]['xy'] = xy
			Aij[nuc][lmp]['r'] = r
			Aij.nodes[lmp]['nuc'] = nuc
			Aij.nodes[lmp]['growing'] = True
			Aij.nodes[lmp]['direction'] = -vec / r

		index_m += n_lmp

	n_node = Aij.number_of_nodes()
	fibre_ends = [i for i in range(n_nuc, n_node) if Aij.degree[i] == 1]
	fibre_grow = [fibre for fibre in fibre_ends if Aij.nodes[fibre]['growing']]
	n_fibres = len(fibre_ends)

	print("No. nodes created = {}".format(n_node))
	print("No. fibres to grow = {}".format(n_fibres))

	it = 0
	total_time = 0
	while len(fibre_grow) > 0:
		start = time.time()

		tot_node_coord = [Aij.nodes[node]['xy'] for node in Aij]
		tot_node_coord = np.stack(tot_node_coord)

		#"""Serial Version
		for fibre in fibre_grow:
			grow(fibre, smoothed, Aij, tot_node_coord, lmp_thresh, theta_thresh, r_thresh)
		#"""

		"""Multi Threading Version
		n_batches = len(fibre_grow) // max_threads + 1
		thread_batches = np.array_split(fibre_grow, n_batches)

		for batch in thread_batches:
			thread_pool = []

			for fibre in batch:

				thread = threading.Thread(target=grow, args=(fibre, smoothed, Aij, tot_node_coord, lmp_thresh, theta_thresh, r_thresh))
				thread.daemon = True
				thread_pool.append(thread)			
		
			for thread in thread_pool: thread.start()
			for thread in thread_pool: thread.join()
		#"""

		n_node = Aij.number_of_nodes()
		fibre_ends = [i for i in range(n_nuc, n_node) if Aij.degree[i] == 1]
		fibre_grow = [fibre for fibre in fibre_ends if Aij.nodes[fibre]['growing']]

		it += 1
		end = time.time()
		total_time += end - start

		print("Iteration {} time = {} s, {} nodes  {}/{} fibres left to grow".format(
			it, round(end - start, 3), n_node, len(fibre_grow), n_fibres))

	for node in Aij.nodes(): Aij.nodes[node]['xy'] = np.array(Aij.nodes[node]['xy'] // scale, dtype=int)
	for edge in Aij.edges(): Aij.edges[edge]['r'] *= 1. / scale

	print("Checking for redundant nodes")
	checking = True
	r2_min = 4

	while checking:
		"Calculate distances between each node"
		node_coord = [Aij.nodes[i]['xy'] for i in Aij.nodes()]
		node_coord = np.stack(node_coord)
		d_coord, r2_coord = distance_matrix(node_coord)

		"Deal with one set of coordinates"
		upper_diag = np.triu_indices(r2_coord.shape[0])
		r2_coord[upper_diag] = r2_min

		"Find nodes in the same location"
		duplicate_nodes = np.where(r2_coord < r2_min)
		checking = (duplicate_nodes[0].size > 0)
		
		"Iterate through each duplicate and transfer edges on to most connected"
		for node1, node2 in zip(*duplicate_nodes):
			if Aij.degree[node1] > Aij.degree[node2]: transfer_edges(Aij, node1, node2)
			elif Aij.degree[node1] + Aij.degree[node2] != 0: transfer_edges(Aij, node2, node1)

		"Remove all nodes with no edges"
		Aij.remove_nodes_from(list(nx.isolates(Aij)))
		mapping = dict(zip(Aij.nodes, np.arange(Aij.number_of_nodes())))
		Aij = nx.relabel_nodes(Aij, mapping)

	"Remove graph components containing either only 1 node with 1 edge or 1 node with more than 1 edge"
	node_remove_list = []
	for i, component in enumerate(nx.connected_components(Aij)):
		subgraph = Aij.subgraph(component)
		edge_count = np.array([subgraph.degree[node] for node in subgraph], dtype=int)
		graph_check = np.sum(edge_count > 1) > 1
		graph_check *= np.sum(edge_count == 1) > 1
		if not graph_check: node_remove_list += list(subgraph.nodes())
	
	Aij.remove_nodes_from(node_remove_list)	

	mapping = dict(zip(Aij.nodes, np.arange(Aij.number_of_nodes())))
	Aij = nx.relabel_nodes(Aij, mapping)
	
	return Aij


def get_edge_list(graph, degree_min=2):

	edge_list = np.empty((0, 2), dtype=int)
	for edge in graph.edges:
		edge = np.array(edge)
		degrees = np.array((graph.degree[edge[0]], graph.degree[edge[1]]))

		degree_check = np.any(degrees != 1)
		degree_check *= np.all(degrees <= degree_min)

		if degree_check:
			order = np.argsort(degrees)
			edge_list = np.concatenate((edge_list, 
					np.expand_dims(edge[order], axis=0)))

	return edge_list


def simplify_network(Aij):


	new_Aij = Aij.copy()
	edge_list = get_edge_list(new_Aij, degree_min=2)

	while edge_list.size > 0:
		for edge in edge_list:
			try: new_Aij = nx.contracted_edge(new_Aij, edge, self_loops=False)
			except (ValueError, KeyError): pass
	
		edge_list = get_edge_list(new_Aij, degree_min=2)

	mapping = dict(zip(new_Aij.nodes, np.arange(new_Aij.number_of_nodes())))
	new_Aij = nx.relabel_nodes(new_Aij, mapping)
	    
	node_coord = [new_Aij.nodes[i]['xy'] for i in new_Aij.nodes()]
	node_coord = np.stack(node_coord)
	d_coord, r2_coord = distance_matrix(node_coord)
	r_coord = np.sqrt(r2_coord)

	for edge in new_Aij.edges:
		new_Aij[edge[0]][edge[1]]['r'] = r_coord[edge[0]][edge[1]]


	return new_Aij


def fibre_assignment(network, angle_thresh=70, verbose=False, min_n=4):

	mapping = dict(zip(network.nodes, np.arange(network.number_of_nodes())))
	network = nx.relabel_nodes(network, mapping)

	node_coord = [network.nodes[i]['xy'] for i in network.nodes()]
	node_coord = np.stack(node_coord)
	edge_count = np.array([network.degree[node] for node in network], dtype=int)

	theta_thresh = np.cos((180-angle_thresh) * np.pi / 180) + 1
	d_coord, r2_coord = distance_matrix(node_coord)

	network_ends = np.argwhere(edge_count == 1).flatten()
	tracing = np.where(edge_count == 1, 1, 1)
	tot_fibres = []

	for n, node in enumerate(np.argsort(edge_count)):

		if tracing[node]:

			fibre = Fibre([node])
			fibre.nodes[node]['xy'] = network.nodes[node]['xy']

			new_nodes = np.array(list(network.adj[node]))
			edge_list = edge_count[new_nodes]
			new_node = new_nodes[np.argsort(edge_list)][-1]
			coord_vec = -d_coord[node][new_node]
			coord_r = network[node][new_node]['r']

			fibre.direction = coord_vec / coord_r
			fibre.fibre_l = coord_r
			fibre.euclid_l = np.sqrt(r2_coord[new_node][node])
			fibre.add_node(new_node, xy= network.nodes[new_node]['xy'])
			fibre.add_edge(node, new_node, 
				r=network[new_node][node]['r'])
			fibre.node_list = list(fibre.nodes)

			if verbose:
				print("Start node = ", node, "  coord: ", node_coord[node])
				print("Next fibre node = ", new_node, "  coord: ", node_coord[new_node])
				print("Fibre length = ", coord_r)
				print("Fibre direction = ", fibre.direction)

			while fibre.growing:

				end_node = fibre.node_list[-1]
				new_connect = np.array(list(network.adj[end_node]))

				if verbose: print("Nodes connected to fibre end = {}".format(new_connect))

				new_connect = numpy_remove(new_connect, fibre.node_list)
				#new_connect = numpy_remove(new_connect, np.argwhere(tracing == 0))
				n_edges = new_connect.shape[0]

				if verbose: 
					print("{} possible candidates for next fibre node: {}".format(n_edges, new_connect))
					print("Coords = ", *node_coord[new_connect])

				if n_edges > 0:
					new_coord_vec = d_coord[end_node][new_connect]
					new_coord_r = np.array([network[end_node][n]['r'] for n in new_connect])

					assert np.all(new_coord_r > 0), print(end_node, new_connect, new_coord_vec, new_coord_r, fibre.node_list)

					cos_the = branch_angles(fibre.direction, new_coord_vec, new_coord_r)

					if verbose: print("Cos theta = ", cos_the)

					try:   
						indices = np.argwhere(cos_the + 1 <= theta_thresh).flatten()
						if verbose: print("Nodes lying in fibre growth direction: ", new_connect[indices])
						straight = (cos_the[indices] + 1).argmin()
						index = indices[straight]

						new_node = new_connect[index]
						coord_vec = - new_coord_vec[index]
						coord_r = new_coord_r[index]

						fibre.direction = coord_vec / coord_r
						fibre.fibre_l += coord_r
						fibre.euclid_l = np.sqrt(r2_coord[node][end_node])
						fibre.add_node(new_node, xy= network.nodes[new_node]['xy'])
						fibre.add_edge(end_node, new_node, 
							r=network[new_node][end_node]['r'])
						fibre.node_list = list(fibre.nodes)
						
						if verbose: 
							print("Next fibre node = ", new_node, "  coord: ", node_coord[new_node])
							print("New fibre length = ", fibre.fibre_l, "(+{})".format(coord_r))
							print("New fibre displacement = ", fibre.euclid_l)
							print("New fibre direction = ", fibre.direction)


					except (ValueError, IndexError):fibre.growing = False
				else: fibre.growing = False
				

			if verbose: print("End of fibre ", node, fibre.node_list)
			if fibre.number_of_nodes() >= min_n: 
				tot_fibres.append(fibre)
				for node in fibre:
					tracing[node] = 0

	return tot_fibres

