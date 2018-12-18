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

from scipy import interpolate
from scipy.misc import derivative
from scipy.ndimage import filters, imread, sobel, distance_transform_edt
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

from skimage import measure, transform, img_as_float, exposure, feature
from skimage.morphology import (square, disk, ball, closing, binary_closing, 
									skeletonize, thin, dilation, erosion, medial_axis)
from skimage.filters import rank, threshold_otsu, try_all_threshold
from skimage.color import label2rgb, rgb2hsv, hsv2rgb, gray2rgb
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from skimage.feature import ORB
from skimage.transform import swirl

from sklearn.decomposition import NMF
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import spectral_clustering

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utilities as ut
import image_tools as it


def print_anis_results(fig_dir, fig_name, tot_q, tot_angle, av_q, av_angle):

	nframe = tot_q.shape[0]
	nxy = tot_q.shape[1]
	print('\n Mean image anistoropy = {:>6.4f}'.format(np.mean(av_q)))
	print('Mean pixel anistoropy = {:>6.4f}\n'.format(np.mean(tot_q)))

	plt.figure()
	plt.hist(av_q, bins='auto', density=True, label=fig_name, range=[0, 1])
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.legend()
	plt.savefig('{}{}_av_aniso_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.hist(av_angle, bins='auto', density=True, label=fig_name, range=[-45, 45])
	plt.xlabel(r'Anisotropy')
	plt.xlim(-45, 45)
	plt.legend()
	plt.savefig('{}{}_av_angle_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	"""
	plt.figure()
	plt.imshow(tot_q[0], cmap='binary_r', interpolation='nearest', origin='lower', vmin=0, vmax=1)
	plt.colorbar()
	plt.savefig('{}{}_anisomap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tot_angle[0], cmap='nipy_spectral', interpolation='nearest', origin='lower', vmin=-45, vmax=45)
	plt.colorbar()
	plt.savefig('{}{}_anglemap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()
	"""
	q_hist = np.zeros(100)
	angle_hist = np.zeros(100)

	for frame in range(nframe):
		q_hist += np.histogram(tot_q[frame].flatten(), bins=100, density=True, range=[0, 1])[0] / nframe
		angle_hist += np.histogram(tot_angle[frame].flatten(), bins=100, density=True, range=[-45, 45])[0] / nframe

	plt.figure()
	plt.title('Anisotropy Histogram')
	plt.plot(np.linspace(0, 1, 100), q_hist, label=fig_name)
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.legend()
	plt.savefig('{}{}_tot_aniso_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Angular Histogram')
	plt.plot(np.linspace(-45, 45, 100), angle_hist, label=fig_name)
	plt.xlabel(r'Angle')
	plt.xlim(-45, 45)
	plt.legend()
	plt.savefig('{}{}_tot_angle_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()


def print_fourier_results(fig_dir, fig_name, angles, fourier_spec, sdi):

	print('\n Modal Fourier Amplitude  = {:>6.4f}'.format(angles[np.argmax(fourier_spec)]))
	print(' Fourier Amplitudes Range   = {:>6.4f}'.format(np.max(fourier_spec)-np.min(fourier_spec)))
	print(' Fourier Amplitudes Std Dev = {:>6.4f}'.format(np.std(fourier_spec)))
	print(' Fourier SDI = {:>6.4f}'.format(sdi))

	print(' Creating Fouier Angle Spectrum figure {}{}_fourier.png'.format(fig_dir, fig_name))
	plt.figure(11)
	plt.title('Fourier Angle Spectrum')
	plt.plot(angles, fourier_spec, label=fig_name)
	plt.xlabel(r'Angle (deg)')
	plt.ylabel(r'Amplitude')
	plt.xlim(-180, 180)
	plt.ylim(0, 1)
	plt.legend()
	plt.savefig('{}{}_fourier.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close('all')


def plot_figures(fig_dir, fig_name, image, anis, angle, energy, tubeness, cmap='viridis'):
	"""
	plot_figures(fig_name, fig_dir, image, anis, angle, energy, tubeness, cmap='viridis')

	Plots a series of figures representing anisotropic analysis of image

	Parameter
	---------
	fig_dir:  string
		Directory of figures to be saved

	fig_name:  string
		Name of figures to be saved

	image:  array_like (float); shape=(n_x, n_y)
		Image under analysis of pos_x and pos_y

	anis:  array_like (float); shape=(n_x, n_y)
		Anisotropy values of image at each pixel

	angle:  array_like (float); shape=(n_x, n_y)
		Angle values of image

	energy:  array_like (float); shape=(n_x, n_y)
		Energy values of image

	"""

	norm_energy = energy / energy.max()

	fig, ax = plt.subplots(figsize=(10, 6))
	plt.imshow(image, cmap=cmap, interpolation='nearest')
	ax.set_axis_off()
	plt.savefig('{}{}.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()
	#"""
	picture = gray2rgb(image)
	hue = (angle + 90) / 180
	saturation = anis / anis.max()
	brightness = image / image.max()
	picture = it.set_HSB(picture, hue, saturation, brightness)

	plt.figure()
	plt.imshow(picture, vmin=0, vmax=1)
	plt.axis("off")
	plt.savefig('{}{}_picture.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()
	#"""
	plt.figure()
	plt.imshow(anis, cmap='binary', interpolation='nearest', vmin=0, vmax=1)
	plt.colorbar()
	plt.savefig('{}{}_anisomap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(angle, cmap='nipy_spectral', interpolation='nearest', vmin=-90, vmax=90)
	plt.colorbar()
	plt.savefig('{}{}_anglemap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(energy, cmap='binary', interpolation='nearest')
	plt.axis("off")
	plt.colorbar()
	plt.savefig('{}{}_energymap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tubeness, cmap='binary', interpolation='nearest')
	plt.axis("off")
	plt.colorbar()
	plt.savefig('{}{}_tubemap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Anisotropy Histogram')
	plt.hist(anis.flatten(), bins=100, density=True, label=fig_name, range=[0, 1])
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.ylim(0, 10)
	plt.legend()
	plt.savefig('{}{}_tot_aniso_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Angular Histogram')
	plt.hist(angle[np.argwhere(norm_energy > 0.02)].flatten(), 
				bins=100, density=True, label=fig_name, range=[-90, 90],
				weights=anis[np.argwhere(norm_energy > 0.02)].flatten())
	plt.xlabel(r'Angle')
	plt.xlim(-90, 90)
	plt.ylim(0, 0.05)
	plt.legend()
	plt.savefig('{}{}_tot_angle_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()


def plot_labeled_figure(fig_dir, fig_name, image, label_image, labels, mode):
	"""
	plot_labeled_figure(fig_dir, fig_name, image, label_image, labels)

	Plots a figure showing identified areas of anisotropic analysis

	Parameter
	---------

	fig_name:  string
		Name of figures to be saved

	fig_dir:  string
		Directory of figures to be saved

	image:  array_like (float); shape=(n_x, n_y)
		Image under analysis of pos_x and pos_y

	label_image:  array_like (int); shape=(n_x, n_y)
		Labelled array with identified anisotropic regions 

	labels:  array_like (int)
		List of labels to plot on figure

	"""
	cluster_image = np.zeros(image.shape)
	rect = []

	for i, n in enumerate(labels):
		region =  measure.regionprops(label_image)[n]
		minr, minc, maxr, maxc = region.bbox
		rect.append(mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
								fill=False, edgecolor='red', linewidth=2))

		indices = np.mgrid[minr:maxr, minc:maxc]
		net_area, network = it.network_area(region.image)
		cluster_image[(indices[0], indices[1])] += network * (i+1)
			
	fig, ax = plt.subplots(figsize=(10, 6))

	image_label_overlay = label2rgb(cluster_image, image=image, bg_label=0)
	ax.imshow(image_label_overlay, cmap='Greys')
	#for r in rect: ax.add_patch(r)
	ax.set_axis_off()
	plt.savefig('{}{}_{}_labels.png'.format(fig_dir, fig_name, mode), bbox_inches='tight')
	plt.close()


def plot_network(fig_dir, fig_name, image, regions, networks):

	BASE_COLOURS = {
	    'b': (0, 0, 1),
	    'g': (0, 0.5, 0),
	    'r': (1, 0, 0),
	    'c': (0, 0.75, 0.75),
	    'm': (0.75, 0, 0.75),
	    'y': (0.75, 0.75, 0),
	    'k': (0, 0, 0)}

	colours = list(BASE_COLOURS.keys())
	

	plt.figure(0, (10,10))
	plt.imshow(image, cmap='Greys')

	for j, Aij in enumerate(networks):

		colour = BASE_COLOURS[colours[j % len(colours)]]
		node_coord = np.stack((Aij.nodes[i]['xy'] for i in Aij.nodes()))
		
		mapping = dict(zip(Aij.nodes, np.arange(Aij.number_of_nodes())))
		Aij = nx.relabel_nodes(Aij, mapping)

		#plt.scatter(node_coord[:,1], node_coord[:,0], c=colour)
		for n, node in enumerate(Aij.nodes):
			for m in list(Aij.adj[node]):
				plt.plot([node_coord[n][1], node_coord[m][1]], 
					 	 [node_coord[n][0], node_coord[m][0]], c=colour)

	plt.savefig('{}{}_networks.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()
