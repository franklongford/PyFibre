"""
ColECM: Collagen ExtraCellular Matrix Simulation
EXPERIMENTAL ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 10/08/2018

Last Modified: 10/08/2018
"""
import sys, os
import numpy as np
import scipy as sp
from contextlib import suppress
import networkx as nx

from skimage import data, measure
from skimage.morphology import convex_hull_image
from skimage.transform import swirl, rescale
from skimage.color import label2rgb, gray2rgb
from skimage.filters import threshold_otsu, hessian
from skimage.restoration import denoise_tv_chambolle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utilities as ut
import image_tools as it
from filters import tubeness
from extraction import FIRE
from graphs import plot_figures, plot_labeled_figure, plot_network


def analyse_image(current_dir, input_file_name, image, size=None, sigma=None, n_clusters=6, ow_anis=False, mode='SHG'):

	cmap = 'viridis'

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'
	image_name = input_file_name

	fig_name = ut.check_file_name(image_name, extension='tif')

	print(' {}'.format(fig_name))

	if not ow_anis and os.path.exists(data_dir + fig_name + '.npy'):
		averages = ut.load_npy(data_dir + fig_name)

	else:
		image = it.prepare_image_shg(image, sigma=sigma, threshold=True, clip_limit=0.01)

		fig, ax = plt.subplots(figsize=(10, 6))
		plt.imshow(image, cmap=cmap, interpolation='nearest')
		ax.set_axis_off()
		plt.savefig('{}{}_orig.png'.format(fig_dir, fig_name), bbox_inches='tight')
		plt.close()

		n_tensor = it.form_nematic_tensor(image, sigma=sigma)
		j_tensor = it.form_structure_tensor(image, sigma=sigma)
		H_tensor = it.form_hessian_tensor(image, sigma=sigma)
		pix_tube = tubeness(image, sigma=sigma)

		"Perform anisotropy analysis on each pixel"

		pix_n_anis, pix_n_angle, pix_n_energy = it.tensor_analysis(n_tensor)
		pix_j_anis, pix_j_angle, pix_j_energy = it.tensor_analysis(j_tensor)
		pix_H_anis, pix_H_angle, pix_H_energy = it.tensor_analysis(H_tensor)

		#plot_figures(fig_dir, fig_name, image, pix_j_anis, pix_j_angle, pix_j_energy, cmap='viridis')

		fig, ax = plt.subplots(figsize=(10, 6))
		plt.imshow(np.where(pix_n_energy > threshold_otsu(pix_n_energy), 1, 0), cmap='Greys', interpolation='nearest')
		plt.colorbar()
		ax.set_axis_off()
		plt.savefig('{}{}_pix_n_energy.png'.format(fig_dir, fig_name), bbox_inches='tight')
		plt.close()
	
		fig, ax = plt.subplots(figsize=(10, 6))
		plt.imshow(np.where(pix_n_energy > threshold_otsu(pix_n_energy), pix_tube, 0), cmap='Greys', interpolation='nearest')
		plt.colorbar()
		ax.set_axis_off()
		plt.savefig('{}{}_pix_tube.png'.format(fig_dir, fig_name), bbox_inches='tight')
		plt.close()

		"Perform anisotropy analysis on whole image"

		img_anis, _ , _ = it.tensor_analysis(np.mean(n_tensor, axis=(0, 1)))
		img_H_anis, _ , _ = it.tensor_analysis(np.mean(H_tensor, axis=(0, 1)))

		"Extract main collagen network using mean curvature"

		(label_image, sorted_areas, regions, networks) = it.network_extraction(image, pix_n_energy, n_clusters)
	
		plot_network(fig_dir, fig_name, image, regions, networks)

		(net_area, net_anis, net_linear, net_cluster,
		fibre_waviness, net_waviness, region_anis, coverage, solidity) = it.network_analysis(label_image, sorted_areas, networks, j_tensor, pix_j_anis)

		plot_labeled_figure(fig_dir, fig_name, image, label_image, sorted_areas, mode)

		clustering = ut.nanmean(net_cluster, weights=net_area)
		anisotropy = ut.nanmean(net_anis, weights=net_area)
		linearity = ut.nanmean(net_linear, weights=net_area)
		pix_anis = ut.nanmean(region_anis, weights=net_area)
		solidity = ut.nanmean(solidity, weights=net_area)
		fibre_waviness = ut.nanmean(fibre_waviness, weights=net_area)
		net_waviness = ut.nanmean(net_waviness, weights=net_area)

		averages = (clustering, linearity, coverage, fibre_waviness, net_waviness, solidity, anisotropy, pix_anis, img_anis)

		ut.save_npy(data_dir + fig_name, averages)

	return averages


def predictor_metric(clus, lin, cover, solid, fibre_waviness, net_waviness, region_anis, pix_anis, img_anis):

	predictor = np.sqrt(fibre_waviness**2 + net_waviness**2 + clus**2 + lin**2 + region_anis**2 + pix_anis**2) / np.sqrt(6)

	return predictor


def analyse_directory(current_dir, input_files, key=None, ow_anis=False):

	print()

	size = 2
	sigma = 0.5

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'

	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	removed_files = []

	for file_name in input_files:
		if not (file_name.endswith('.tif')): removed_files.append(file_name)
		elif (file_name.find('display') != -1): removed_files.append(file_name)
		elif (file_name.find('AVG') == -1): removed_files.append(file_name)
		
		if key != None:
			if (file_name.find(key) == -1) and (file_name not in removed_files): 
				removed_files.append(file_name)

	for file_name in removed_files: input_files.remove(file_name)

	SHG_files = []
	PL_files = []
	for file_name in input_files:
		if (file_name.find('SHG') != -1): SHG_files.append(file_name)
		elif (file_name.find('PL') != -1): PL_files.append(file_name)

	input_list = [SHG_files, PL_files]
	modes = ['SHG']#, 'PL']

	for n, mode in enumerate(modes):
		input_files = input_list[n]

		ske_clus = np.zeros(len(input_files))
		ske_path = np.zeros(len(input_files))
		ske_solid = np.zeros(len(input_files))
		ske_lin = np.zeros(len(input_files))
		ske_cover = np.zeros(len(input_files))
		mean_img_anis = np.zeros(len(input_files))
		mean_pix_anis = np.zeros(len(input_files))
		mean_ske_anis = np.zeros(len(input_files))
		fibre_waviness = np.zeros(len(input_files))
		net_waviness = np.zeros(len(input_files))

		for i, input_file_name in enumerate(input_files):
			image = it.load_tif(input_file_name)
			#image = rescale(image, 2)
			res = analyse_image(current_dir, input_file_name, image, size=size, 
								sigma=sigma, ow_anis=ow_anis, mode=mode)
			(ske_clus[i], ske_lin[i], ske_cover[i], fibre_waviness[i], net_waviness[i], ske_solid[i],
				mean_ske_anis[i], mean_pix_anis[i], mean_img_anis[i]) = res

			print(' Network Clustering = {:>6.4f}'.format(ske_clus[i]))
			print(' Network Linearity = {:>6.4f}'.format(ske_lin[i]))
			print(' Network Coverage = {:>6.4f}'.format(ske_cover[i]))
			print(' Network Solidity = {:>6.4f}'.format(ske_solid[i]))
			print(' Network Anistoropy = {:>6.4f}'.format(mean_ske_anis[i]))
			print(' Network Waviness = {:>6.4f}'.format(net_waviness[i]))
			print(' Av. Fibre Waviness = {:>6.4f}'.format(fibre_waviness[i]))
			print(' Total Pixel anistoropy = {:>6.4f}'.format(mean_pix_anis[i]))
			print(' Total Image anistoropy = {:>6.4f}\n'.format(mean_img_anis[i]))

		x_labels = [ut.check_file_name(image_name, extension='tif') for image_name in input_files]
		col_len = len(max(x_labels, key=len))

		predictor = predictor_metric(ske_clus, ske_lin, ske_cover, ske_solid, fibre_waviness, net_waviness,
						mean_ske_anis, mean_pix_anis, mean_img_anis)

		for i, file_name in enumerate(x_labels): 
			if np.isnan(predictor[i]):
				predictor = np.array([x for j, x in enumerate(predictor) if j != i])
				x_labels.remove(file_name)

		ut.bubble_sort(x_labels, predictor)
		x_labels = x_labels[::-1]
		predictor = predictor[::-1]
		#sorted_predictor = np.argsort(predictor)

		print("Order of total predictor:")
		print(' {:{col_len}s} | {:10s} | {:10s}'.format('', 'Predictor', 'Order', col_len=col_len))
		print("_" * 75)

		for i, name in enumerate(x_labels):
			print(' {:{col_len}s} | {:10.3f} | {:10d}'.format(name, predictor[i], i, col_len=col_len))

		print('\n')
