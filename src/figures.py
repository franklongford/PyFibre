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

from skimage import draw
from skimage.transform import rotate
from skimage.color import label2rgb, grey2rgb, rgb2grey, rgb2hsv, hsv2rgb

from scipy.ndimage.filters import gaussian_filter

import utilities as ut
from filters import form_structure_tensor
from analysis import tensor_analysis


def create_figure(image, filename, figsize=(10, 10), ext='png', cmap='viridis'):

	import matplotlib.pyplot as plt

	plt.figure(figsize=figsize)
	if image.ndim == 2: plt.imshow(image, cmap=cmap)
	else: plt.imshow(image)
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(filename + '.' + ext)


def set_HSB(image, hue, saturation=1, brightness=1):
	""" Add color of the given hue to an greyscale image.

	By default, set the saturation to 1 so that the colors pop!
	"""
	rgb = grey2rgb(image)
	hsv = rgb2hsv(rgb)

	hsv[..., 0] = hue
	hsv[..., 1] = saturation
	hsv[..., 2] = brightness

	return hsv2rgb(hsv)


def create_tensor_image(image, N=120):

	"Form nematic and structure tensors for each pixel"
	j_tensor = form_structure_tensor(image, sigma=1.0)

	"Perform anisotropy analysis on each pixel"
	pix_j_anis, pix_j_angle, pix_j_energy = tensor_analysis(j_tensor)

	hue = (pix_j_angle + 90) / 180
	saturation = pix_j_anis / pix_j_anis.max()
	brightness = image / image.max()

	"Make circular test image"
	image_grid = np.mgrid[:N, :N]
	for i in range(2): 
		image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
		image_grid[i] = np.fft.fftshift(image_grid[i])

	image_radius = np.sqrt(np.sum(image_grid**2, axis=0))
	image_rings = np.sin(4 * np.pi * image_radius / N ) * np.cos(4 * np.pi * image_radius / N)

	j_tensor = form_structure_tensor(image_rings, sigma=1.0)
	pix_j_anis, pix_j_angle, pix_j_energy = tensor_analysis(j_tensor)

	pix_j_angle = rotate(pix_j_angle, 90)[ : N // 2  ]
	pix_j_anis = pix_j_anis[ : N // 2 ]
	pix_j_energy = np.where(image_radius < (N / 2), pix_j_energy, 0)[ : N // 2 ]

	hue[- N // 2 : image.shape[0], : N] = (pix_j_angle + 90) / 180
	saturation[- N // 2 : image.shape[0], : N] = pix_j_anis / pix_j_anis.max()
	brightness[- N // 2 : image.shape[0], : N] = pix_j_energy / pix_j_energy.max()

	"Form structure tensor image"
	rgb_image = set_HSB(image, hue, saturation, brightness)

	return rgb_image


def create_region_image(image, regions):
	"""
	Plots a figure showing identified regions

	Parameter
	---------

	image:  array_like (float); shape=(n_x, n_y)
		Image under analysis of pos_x and pos_y

	label_image:  array_like (int); shape=(n_x, n_y)
		Labelled array with identified anisotropic regions 

	regions:  list (skimage.region)
		List of segmented regions

	"""
	
	image /= image.max()
	label_image = np.zeros(image.shape, dtype=int)
	label = 1
	
	for region in regions:
		minr, minc, maxr, maxc = region.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		label_image[(indices[0], indices[1])] += region.image * label
		label += 1

	image_label_overlay = label2rgb(label_image, image=image, bg_label=0,
					image_alpha=0.99, alpha=0.25, bg_color=(0, 0, 0))

	return image_label_overlay


def create_network_image(image, networks, c_mode=0):

	BASE_COLOURS = {
	    'b': (0, 0, 1),
	    'g': (0, 0.5, 0),
	    'r': (1, 0, 0),
	    'c': (0, 0.75, 0.75),
	    'm': (0.75, 0, 0.75),
	    'y': (0.75, 0.75, 0),
	    'k': (0, 0, 0)}

	colours = list(BASE_COLOURS.keys())
	
	rgb_image = grey2rgb(image)
	
	for j, network in enumerate(networks):

		if c_mode == 0: colour = BASE_COLOURS['r']
		else: colour = BASE_COLOURS[colours[j % len(colours)]]

		node_coord = [network.nodes[i]['xy'] for i in network]
		node_coord = np.stack((node_coord))
		
		mapping = zip(network.nodes, np.arange(network.number_of_nodes()))
		mapping_dict = dict(mapping)		

		for n, node1 in enumerate(network):
			for node2 in list(network.adj[node1]):
				m = mapping_dict[node2]
				rr, cc, val = draw.line_aa(node_coord[m][0], node_coord[m][1],
						   node_coord[n][0], node_coord[n][1])
				
				for i, c in enumerate(colour): rgb_image[rr, cc, i] = c * val * 255.9999

	return rgb_image

		
