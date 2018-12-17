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

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import coo_matrix, csr_matrix, csgraph

from skimage.feature import (structure_tensor, hessian_matrix, hessian_matrix_eigvals)
from skimage.filters import threshold_otsu
from skimage.morphology import square, local_maxima, binary_erosion

import networkx as nx

from utilities import ring, numpy_remove, clear_border


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

	if coord.shape[0] <= 1: return coord

	thresh = np.sqrt(2 * thresh**2)
	r_coord = cdist(coord, coord)

	del_coord = np.argwhere((r_coord <= thresh) - np.identity(coord.shape[0]))
	del_coord = del_coord[np.arange(0, del_coord.shape[0], 2)]
	indices = np.stack((values[del_coord[:,0]], values[del_coord[:,1]])).argmax(axis=0)
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
        branch_coord = reduce_coord(branch_coord, image[branch_coord[:,0], branch_coord[:,1]])
        
        n_branch = branch_coord.shape[0]
        branch_vector = np.tile(coord, (n_branch, 1)) - branch_coord
        branch_r = np.sqrt(np.sum(branch_vector**2, axis=1))
        
        return branch_coord, branch_vector, branch_r


class Fibre:

	def __init__(self, index, direction=0, growing=True):

		self.nodes = np.array([index])
		self.direction = direction
		self.growing = growing

	    
	def add_node(self, index, direction):

	    self.nodes = np.concatenate((self.nodes, [index]))
	    self.direction = direction

	    
	def grow(self, image, Aij, tot_node_coord, lmp_thresh, theta_thresh, r_thresh):

	    start_coord = Aij.nodes[self.nodes[0]]['xy']
	    end_coord = Aij.nodes[self.nodes[-1]]['xy']
	    
	    ring_filter = ring(np.zeros(image.shape), end_coord, np.arange(2, 4), 1)
	    branch_coord, branch_vector, branch_r = new_branches(image, end_coord, 
	                                                         ring_filter, lmp_thresh)
	    n_branch = branch_coord.shape[0]
	    dir_vector = np.tile(self.direction, (n_branch, 1))
	    dir_r = np.ones(n_branch)

	    combined_vector = np.hstack((branch_vector, dir_vector)).reshape(n_branch*2, 2)
	    combined_r = np.column_stack((branch_r, dir_r)).flatten()
	    cos_the = cos_sin_theta_2D(combined_vector, combined_r)
	    
	    indices = np.argwhere(abs(cos_the + 1) <= theta_thresh)

	    if indices.size == 0: 
	        self.growing = False
	        return

	    branch_coord = branch_coord[indices]
	    branch_vector = branch_vector[indices]
	    branch_r = branch_r[indices]

	    close_nodes, _ = check_2D_arrays(tot_node_coord, branch_coord, 2)
	    close_nodes = numpy_remove(close_nodes, self.nodes)

	    if close_nodes.size != 0:

	        new_end = close_nodes.min()

	        new_end_coord = Aij.nodes[new_end]['xy']
	        Aij.add_edge(self.nodes[-1], new_end)
	        Aij[self.nodes[-1]][new_end]['r'] = np.sqrt(((new_end_coord - end_coord)**2).sum())

	        new_dir_vector = new_end_coord - start_coord
	        new_dir_r = np.sqrt((new_dir_vector**2).sum())

	        self.growing = False
	        self.add_node(new_end, (new_dir_vector / new_dir_r))

	    else:
	        index = branch_r.argmax()

	        new_end_coord = branch_coord[index].flatten()
	        new_end_vector = new_end_coord - Aij.nodes[self.nodes[-2]]['xy']
	        new_end_r = np.sqrt((new_end_vector**2).sum())

	        new_dir_vector = new_end_coord - start_coord
	        new_dir_r = np.sqrt((new_dir_vector**2).sum())

	        if new_end_r >= r_thresh:

	            new_end = Aij.number_of_nodes()
	            Aij.add_node(new_end)
	            Aij.add_edge(self.nodes[-1], new_end)
	            Aij.nodes[new_end]['xy'] = new_end_coord
	            Aij[self.nodes[-1]][new_end]['r'] = np.sqrt(((new_end_coord - end_coord)**2).sum())

	            self.add_node(new_end, (new_dir_vector / new_dir_r))

	        else: 
	            Aij.nodes[self.nodes[-1]]['xy'] = new_end_coord
	            Aij[self.nodes[-1]][self.nodes[-2]]['r'] = new_end_r

	            self.direction = (new_dir_vector / new_dir_r)

def FIRE(image, sigma = 1.5, lambda_=0.5, nuc_thresh=2, lmp_thresh=0.2, 
             angle_thresh=70, r_thresh=8):

    "Prepare input image to gain distance matrix of foreground from background"
    cleared = clear_border(image)
    threshold = cleared > threshold_otsu(cleared)
    distance = distance_transform_edt(threshold)
    smoothed = gaussian_filter(distance, sigma=sigma)

    "Set distance and angle thresholds for fibre iterator"
    nuc_thresh = np.min([nuc_thresh, 0.3 * smoothed.max()])
    lmp_thresh = np.min([lmp_thresh, 0.1 * smoothed.max()])
    theta_thresh = np.cos((180-angle_thresh) * np.pi / 180) + 1

    print("Maximum distance = {}".format(smoothed.max()))
    print("Using thresholds:\n nuc = {} pix  lmp = {} pix\n angle = {} deg".format(
                    nuc_thresh, lmp_thresh, angle_thresh))

    "Get global maxima for smoothed distance matrix"
    maxima = local_maxima(smoothed, selem=square(9))
    nuc_node_coord = reduce_coord(np.argwhere(maxima * smoothed >= nuc_thresh),
                            smoothed[np.where(maxima * smoothed >= nuc_thresh)], 10)

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

        ring_filter = ring(np.zeros(smoothed.shape), nuc_coord, [4], 1)
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
            
            tot_fibres.append(Fibre(nuc))
            tot_fibres[-1].add_node(lmp, -vec / r)

        index_m += n_lmp

    n_node = Aij.number_of_nodes()
    n_fibres = len(tot_fibres)
    fibre_grow = [fibre.growing for fibre in tot_fibres]

    print("No. nodes created = {}".format(n_node))
    print("No. fibres to grow = {}".format(n_fibres))

    it = 0
    while np.any(fibre_grow):
        n_node = Aij.number_of_nodes()
    
        print("Iteration {}, {} nodes  {}/{} fibres left to grow".format(
                it, n_node, int(np.sum(fibre_grow)), n_fibres))

        tot_node_coord = np.stack((Aij.nodes[i]['xy'] for i in Aij.nodes()))

        for fibre in tot_fibres:
            
            if fibre.growing:
                fibre.grow(smoothed, Aij, tot_node_coord, lmp_thresh, 
                                 theta_thresh, r_thresh)
                
        fibre_grow = [fibre.growing for fibre in tot_fibres]

        it += 1

    return Aij


def adj_analysis(Aij, angle_thresh=70):
    
    node_coord = np.stack((Aij.nodes[i]['xy'] for i in Aij.nodes()))
    theta_thresh = np.cos((180-angle_thresh) * np.pi / 180) + 1
    d_coord, r2_coord = distance_matrix(node_coord)
    waviness = np.empty((0,), dtype='float64')
    edge_count = np.array([Aij.degree[node] for node in Aij.nodes], dtype=int)
    tracing = np.where(edge_count == 1, 1, 0)

    for n, node in enumerate(Aij.nodes):
        if tracing[node]:
            old_node = node
            new_node = np.array(list(Aij.adj[node]))[0]#np.argwhere(Aij[node]).flatten()[0]
            fibre = np.array([old_node, new_node])
            coord_vec = -d_coord[old_node][new_node]
            coord_r = Aij[node][new_node]['r']#np.sqrt(r2_coord[node][new_node])
            fibre_l = coord_r
            #print("Start node = ", old_node, node_coord[old_node], coord_r)
            #print("Next fibre node = ", new_node, node_coord[new_node])
            #print("Fibre length = ", coord_r)
            while tracing[node]:
                
                new_connect = np.array(list(Aij.adj[new_node]))#np.argwhere(Aij[new_node]).flatten()
                new_connect = numpy_remove(new_connect, [old_node])
                n_edges = new_connect.shape[0]
                #print("New connect = ", new_connect, n_edges)
                
                if n_edges > 1:
                    coord_vec = np.tile(coord_vec, (n_edges, 1))
                    coord_r = np.repeat(coord_r, n_edges)
                    new_coord_vec = d_coord[new_node][new_connect]
                    new_coord_r = np.array([Aij[new_node][n]['r'] for n in new_connect])

                    combined_vector = np.hstack((new_coord_vec, 
                                                 coord_vec)).reshape(n_edges*2, 2)
                    combined_r = np.column_stack((new_coord_r, coord_r)).flatten()
                    
                    cos_the = cos_sin_theta_2D(combined_vector, combined_r)
                    #print("Cos theta = ", cos_the)
                    try:   
                        indices = np.argwhere(cos_the + 1 <= theta_thresh).flatten()
                        #print("Possible fibre nodes ", indices)
                        straight = (cos_the[indices] + 1).argmin()
                        index = indices[straight]
                        #print(index, indices[straight], cos_the[index], new_coord_r[index])
                        
                        old_node = new_node
                        coord_vec = - new_coord_vec[index]
                        coord_r = new_coord_r[index]
                        fibre_l += coord_r
                        new_node = new_connect[index]
                        #print("Next fibre node = ", new_node, node_coord[new_node])
                        #print("New fibre length = ", fibre_l, coord_r)

                    except (ValueError, IndexError): 
                        tracing[node] = 0
                        tracing[new_node] = 0
                        
                elif n_edges == 1:
                    old_node = new_node
                    new_node = new_connect[0]
                    #print("Next fibre node = ", new_node, node_coord[new_node])
                    coord_vec = -d_coord[old_node][new_node]
                    coord_r = Aij[old_node][new_node]['r']#np.sqrt(r2_coord[old_node][new_node])
                    fibre_l += coord_r
                    #print("New fibre length = ", fibre_l, coord_r, np.sqrt(r2_coord[old_node][new_node]))
                    
                else: 
                    tracing[node] = 0
                    tracing[new_node] = 0
                
                if tracing[node]: fibre = np.concatenate((fibre, [new_node]))
            
            #print("End of fibre ", node, fibre)    
            if fibre.shape[0] > 3:
                euclid_dist = np.sqrt(r2_coord[fibre[0]][fibre[-1]])
                #print("Terminals = ", node_coord[fibre[0]], node_coord[fibre[-1]],
                #					  node_coord[fibre[0]] - node_coord[fibre[-1]],
                #					  d_coord[fibre[0]][fibre[-1]],
                #					  d_coord[fibre[0]][fibre[-1]]**2)
                print("Length", fibre_l, "Displacement", euclid_dist)

                waviness = np.concatenate((waviness, [euclid_dist / fibre_l]))

    return waviness.mean()

