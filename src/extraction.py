"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import numpy as np
import sys
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
from filters import hysteresis


def check_2D_arrays(array1, array2, thresh=1):

	array1_mat = np.tile(array1, (1, array2.shape[0]))\
	                .reshape(array1.shape[0], array2.shape[0], 2)    
	array2_mat = np.tile(array2, (array1.shape[0], 1))\
	                .reshape(array1.shape[0], array2.shape[0], 2)

	diff = np.sum((array1_mat - array2_mat)**2, axis=2)
	array1_indices = np.argwhere(diff <= thresh**2)[:,0]
	array2_indices = np.argwhere(diff <= thresh**2)[:,1]

	return array1_indices, array2_indices


def distance_matrix(node_coord):

	node_coord_matrix = np.tile(node_coord, (node_coord.shape[0], 1))\
	                    .reshape(node_coord.shape[0], node_coord.shape[0], 2)
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
			Aij.nodes[target]['fibres'].update(Aij.nodes[source]['fibres'])
			Aij.nodes[source]['fibres'] = set()

class Fibre:

	def __init__(self, index, nodes, direction=0, growing=True):

		self.index = index
		self.nodes = np.array([nodes])
		self.direction = direction
		self.growing = growing

	    
	def add_node(self, nodes, direction):

		self.nodes = np.concatenate((self.nodes, [nodes]))
		self.direction = direction

	    
def grow(fibre, image, Aij, tot_node_coord, lmp_thresh, theta_thresh, r_thresh):
	"""
	Grow fibre object along network

	Parameters
	----------
	
	fibre: Fibre
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

	start_coord = Aij.nodes[fibre.nodes[0]]['xy']
	end_coord = Aij.nodes[fibre.nodes[-1]]['xy']

	ring_filter = ring(np.zeros(image.shape), end_coord, np.arange(2, 3), 1)
	branch_coord, branch_vector, branch_r = new_branches(image, end_coord, 
			                                 ring_filter, lmp_thresh)
	cos_the = branch_angles(fibre.direction, branch_vector, branch_r)
	indices = np.argwhere(abs(cos_the + 1) <= theta_thresh)

	if indices.size == 0: 
		fibre.growing = False
		if Aij[fibre.nodes[-1]][fibre.nodes[-2]]['r'] <= 2:
			transfer_edges(Aij, fibre.nodes[-1], fibre.nodes[-2])
		return

	branch_coord = branch_coord[indices]
	branch_vector = branch_vector[indices]
	branch_r = branch_r[indices]

	close_nodes, _ = check_2D_arrays(tot_node_coord, branch_coord, 2)
	close_nodes = numpy_remove(close_nodes, fibre.nodes)

	if close_nodes.size != 0:

		new_end = close_nodes.min()

		end_coord = Aij.nodes[fibre.nodes[-2]]['xy']
		new_end_coord = Aij.nodes[new_end]['xy']

		transfer_edges(Aij, fibre.nodes[-1], new_end)

		new_dir_vector = new_end_coord - start_coord
		new_dir_r = np.sqrt((new_dir_vector**2).sum())

		fibre.growing = False
		fibre.add_node(new_end, (new_dir_vector / new_dir_r))

	else:
		index = branch_r.argmax()

		new_end_coord = branch_coord[index].flatten()
		new_end_vector = new_end_coord - Aij.nodes[fibre.nodes[-2]]['xy']
		new_end_r = np.sqrt((new_end_vector**2).sum())

		new_dir_vector = new_end_coord - start_coord
		new_dir_r = np.sqrt((new_dir_vector**2).sum())

		if new_end_r >= r_thresh:

			new_end = Aij.number_of_nodes()
			Aij.add_node(new_end)
			Aij.add_edge(fibre.nodes[-1], new_end)
			Aij.nodes[new_end]['xy'] = new_end_coord
			Aij.nodes[new_end]['fibres'] = set({fibre.index})
			Aij[fibre.nodes[-1]][new_end]['r'] = np.sqrt(((new_end_coord - end_coord)**2).sum())

			fibre.add_node(new_end, (new_dir_vector / new_dir_r))

		else: 
			Aij.nodes[fibre.nodes[-1]]['xy'] = new_end_coord
			Aij[fibre.nodes[-1]][fibre.nodes[-2]]['r'] = new_end_r

			fibre.direction = (new_dir_vector / new_dir_r)


def FIRE(image, scale=1, alpha=0.75, sigma=0.5, nuc_thresh=2, nuc_rad=15, lmp_thresh=0.15, 
             angle_thresh=70, r_thresh=9, max_threads=8):
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

	alpha *= scale
	sigma *= scale

	cleared = clear_border(image_scale)
	threshold = hysteresis(cleared, alpha=alpha)
	cleaned = remove_small_objects(threshold)
	distance = distance_transform_edt(cleaned)
	smoothed = gaussian_filter(distance, sigma=sigma)

	"Set distance and angle thresholds for fibre iterator"
	nuc_thresh = np.min([nuc_thresh, 2E-1 / scale**2 * smoothed.max()])
	lmp_thresh = np.min([lmp_thresh, 5E-3 / scale**2 * smoothed[np.nonzero(smoothed)].mean()])
	theta_thresh = np.cos((180-angle_thresh) * np.pi / 180) + 1
	r_thresh = int(r_thresh * scale)

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
	tot_fibres = []

	Aij = nx.Graph()
	Aij.add_nodes_from(np.arange(n_nuc))

	"Iterate through nucleation points"
	index_m = n_nuc
	for nuc, nuc_coord in enumerate(nuc_node_coord):
		Aij.nodes[nuc]['xy'] = nuc_coord
		Aij.nodes[nuc]['fibres'] = set({})

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
			Aij.nodes[nuc]['fibres'].add(lmp)
			Aij.nodes[lmp]['fibres'] = set({lmp}) 

			tot_fibres.append(Fibre(lmp, nuc))
			tot_fibres[-1].add_node(lmp, -vec / r)

		index_m += n_lmp

	n_node = Aij.number_of_nodes()
	n_fibres = len(tot_fibres)
	fibre_grow = [fibre.growing for fibre in tot_fibres]

	print("No. nodes created = {}".format(n_node))
	print("No. fibres to grow = {}".format(n_fibres))

	it = 0
	total_time = 0
	while np.any(fibre_grow):
		start = time.time()
		n_node = Aij.number_of_nodes()

		tot_node_coord = [Aij.nodes[i]['xy'] for i in Aij.nodes()]
		tot_node_coord = np.stack(tot_node_coord)
		fibre_indices = np.argwhere(fibre_grow).flatten()

		"""Serial Version
		for fibre in fibre_indices:
			grow(tot_fibres[fibre], smoothed, Aij, tot_node_coord, lmp_thresh, theta_thresh, r_thresh)
		"""

		#"""Multi Threading Version
		n_batches = fibre_indices.size // max_threads + 1
		thread_batches = np.array_split(fibre_indices, n_batches)

		for batch in thread_batches:
			thread_pool = []

			for fibre in batch:
				thread = threading.Thread(target=grow, args=(tot_fibres[fibre], smoothed, Aij, tot_node_coord, lmp_thresh, theta_thresh, r_thresh))
				thread.daemon = True
				thread_pool.append(thread)			
		
			for thread in thread_pool: thread.start()
			for thread in thread_pool: thread.join()
		#"""

		fibre_grow = [fibre.growing for fibre in tot_fibres]
		it += 1
		end = time.time()
		total_time += end - start

		print("Iteration {} time = {} s, {} nodes  {}/{} fibres left to grow".format(
			it, round(end - start, 3), n_node, int(np.sum(fibre_grow)), n_fibres))

	print("Checking for redundant nodes")

	"Calculate distances between each node"
	node_coord = [Aij.nodes[i]['xy'] for i in Aij.nodes()]
	node_coord = np.stack(node_coord)
	d_coord, r2_coord = distance_matrix(node_coord)

	"Deal with one set of coordinates"
	upper_diag = np.triu_indices(r2_coord.shape[0])
	r2_coord[upper_diag] = 1

	"Find nodes in the same location"
	duplicate_nodes = np.where(r2_coord == 0)

	"Iterate through each duplicate and transfer edges on to most connected"
	for node1, node2 in zip(*duplicate_nodes):
		if Aij.degree[node1] > Aij.degree[node2]: transfer_edges(Aij, node1, node2)
		else: transfer_edges(Aij, node2, node1)

	"Remove all nodes with no edges"
	Aij.remove_nodes_from(list(nx.isolates(Aij)))
	mapping = dict(zip(Aij.nodes, np.arange(Aij.number_of_nodes())))
	Aij = nx.relabel_nodes(Aij, mapping)

	for node in Aij.nodes(): Aij.nodes[node]['xy'] = np.array(Aij.nodes[node]['xy'] // scale, dtype=int)
	for edge in Aij.edges(): Aij.edges[edge]['r'] *= 1. / scale

	print(f"TOTAL TIME = {round(total_time, 3)} s")

	return Aij


def adj_analysis(Aij, angle_thresh=70, verbose=False):

	mapping = dict(zip(Aij.nodes, np.arange(Aij.number_of_nodes())))
	Aij = nx.relabel_nodes(Aij, mapping)
	network_waviness = np.empty((0,), dtype='float64')

	#for component in nx.connected_components(Aij):
	#subgraph = Aij.subgraph(component)

	"""
	edge_nodes = np.array([node for node in Aij.nodes if Aij.degree[node] == 1], dtype=int)
	
	for n, node1 in enumerate(edge_nodes):
		for m, node2 in enumerate(edge_nodes[:n]):
			shortest_path_r = nx.shortest_path_length(Aij, source=node1, target=node2, weight='r')
			euclidean_r = np.sqrt(((Aij.nodes[node1]['xy'] - Aij.nodes[node2]['xy'])**2).sum())
			network_waviness = np.concatenate((network_waviness, [euclidean_r / shortest_path_r])) 

	#"""

	node_coord = [Aij.nodes[i]['xy'] for i in Aij.nodes()]
	node_coord = np.stack(node_coord)
	edge_count = np.array([Aij.degree[node] for node in Aij.nodes], dtype=int)
	theta_thresh = np.cos((180-angle_thresh) * np.pi / 180) + 1
	d_coord, r2_coord = distance_matrix(node_coord)
	fibre_waviness = np.empty((0,), dtype='float64')

	tracing = np.where(edge_count == 1, 1, 0)
	tot_fibres = []

	for n, node in enumerate(Aij.nodes):
		if tracing[node]:
			fibre = Fibre(n, node)
			tracing[node] = 0

			new_node = np.array(list(Aij.adj[node]))[0]
			coord_vec = -d_coord[node][new_node]
			coord_r = Aij[node][new_node]['r']
			direction = coord_vec / coord_r

			fibre_l = coord_r
			if verbose:
				print("Start node = ", node, "  coord: ", node_coord[node])
				print("Next fibre node = ", new_node, "  coord: ", node_coord[new_node])
				print("Fibre length = ", coord_r)
				print("Fibre direction = ", fibre.direction)

			while fibre.growing:
		
				fibre.add_node(new_node, direction)
				new_connect = np.array(list(Aij.adj[fibre.nodes[-1]]))#np.argwhere(Aij[new_node]).flatten()

				if verbose: print("Nodes connected to fibre end = {}".format(new_connect))

				new_connect = numpy_remove(new_connect, fibre.nodes)
				n_edges = new_connect.shape[0]

				if verbose: 
					print("{} possible candidates for next fibre node: {}".format(n_edges, new_connect))
					print("Coords = ", *node_coord[new_connect])
	
				if n_edges > 1:
					new_coord_vec = d_coord[new_node][new_connect]
					new_coord_r = np.array([Aij[new_node][n]['r'] for n in new_connect])

					assert np.all(new_coord_r > 0), print(new_node, new_connect, new_coord_vec, new_coord_r, fibre.nodes)

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
						direction = coord_vec / coord_r

						fibre_l += coord_r
						
						if verbose: 
							print("Next fibre node = ", new_node, "  coord: ", node_coord[new_node])
							print("New fibre length = ", fibre_l, "(+{})".format(coord_r))
							print("New fibre direction = ", fibre.direction)


					except (ValueError, IndexError):
						fibre.growing = False
						tracing[fibre.nodes[-1]] = 0
			
				elif n_edges == 1:
					new_node = new_connect[0]
					coord_vec = -d_coord[fibre.nodes[-1]][new_node]
					coord_r = Aij[fibre.nodes[-1]][new_node]['r']

					assert coord_r > 0, print(new_node, coord_vec, coord_r, fibre.nodes)#np.sqrt(r2_coord[old_node][new_node])
					direction = coord_vec / coord_r

					fibre_l += coord_r

					if verbose: 
						print("New fibre length = ", fibre_l, "(+{})".format(coord_r))
						print("New fibre direction = ", direction)
				    
				else:
					fibre.growing = False
					tracing[fibre.nodes[-1]] = 0
		    
			if verbose: print("End of fibre ", node, fibre.nodes)
    
			if fibre.nodes.size > 3:
				euclid_dist = np.sqrt(r2_coord[fibre.nodes[0]][fibre.nodes[-1]])
				
				if verbose:
					print("Terminals = ", node_coord[fibre.nodes[0]], node_coord[fibre.nodes[-1]],
										  node_coord[fibre.nodes[0]] - node_coord[fibre.nodes[-1]],
										  d_coord[fibre.nodes[0]][fibre.nodes[-1]],
										  d_coord[fibre.nodes[0]][fibre.nodes[-1]]**2)
					print("Length", fibre_l, "Displacement", euclid_dist, "\n")

				fibre_waviness = np.concatenate((fibre_waviness, [euclid_dist / fibre_l]))

	#"""

	return  np.nanmean(fibre_waviness)#, fibre_waviness.mean()
	#return  network_waviness.mean()#, fibre_waviness.mean()

