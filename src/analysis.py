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
import pandas as pd
import networkx as nx

from skimage import data, measure, exposure
from skimage.transform import rescale
from skimage.filters import threshold_otsu, hessian
from skimage.restoration import estimate_sigma

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utilities as ut
import image_tools as it
from filters import tubeness
from extraction import FIRE


class NoiseError(Exception):
    
    def __init__(self, noise, thresh):

    	self.noise = noise
    	self.thresh = thresh
    	self.message = "Image too noisy ({} > {})".format(noise, thresh)


def get_snr_estimates(image, sigma, guess):

	clip_limit = [guess - 0.001, guess, guess + 0.001]
	snr = []

	for cl in clip_limit: 
		img = it.preprocess_image(image, sigma=sigma, threshold=True, clip_limit=cl)
		snr.append(get_snr(img))

	d_cl = clip_limit[-1] - clip_limit[0]
	d_snr = snr[-1] - snr[0]
	snr_grad = d_snr / d_cl

	first_derivative = snr_grad

	d_cl = [clip_limit[1] - clip_limit[0], clip_limit[2] - clip_limit[1]]
	d_snr = [snr[1] - snr[0], snr[2] - snr[1]]
	snr_grad = [d_snr[0] / d_cl[0], d_snr[1] / d_cl[1]]

	second_derivative = 0.5 * (snr_grad[-1] - snr_grad[0]) / (clip_limit[-1] - clip_limit[0])

	return snr[1], first_derivative, second_derivative

def get_snr(image):

	noise = estimate_sigma(image, multichannel=False, average_sigmas=True)
	signal = image.sum() / np.count_nonzero(image)

	return signal / noise


def optimise_equalisation(image, sigma, guess=0.1, alpha = 1.0, precision = 2E-1, max_it=100):

	iteration = 1
	clip_limit = [guess]
	snr = []
	snr_grad = []

	img = it.preprocess_image(image, sigma=sigma, threshold=True, clip_limit=clip_limit[-1])
	snr_n, d_snr, dd_snr = get_snr_estimates(image, sigma, clip_limit[-1])
	snr.append(snr_n)
	snr_grad.append(d_snr)

	gamma = alpha * d_snr / dd_snr
	print(iteration, d_snr, dd_snr, clip_limit[-1], snr[-1], snr_grad[-1], gamma)
	clip_limit.append(clip_limit[-1] + gamma)
	
	while True:
		img = it.preprocess_image(image, sigma=sigma, threshold=True, clip_limit=clip_limit[-1])
		snr_n, d_snr, dd_snr = get_snr_estimates(image, sigma, clip_limit[-1])
		snr.append(snr_n)
		snr_grad.append(d_snr)
		
		gamma = d_snr * (clip_limit[-1] - clip_limit[-2]) * (snr_grad[-1] - snr_grad[-2]) / abs(snr_grad[-1] - snr_grad[-2])**2
		new_clip_limit = clip_limit[-1] + gamma 
		check = (new_clip_limit >= 0) * (new_clip_limit < 0.5) * (iteration <= max_it) * (abs(d_snr) >= precision)

		if not check:

			#plt.figure(0)
			#plt.scatter(clip_limit, snr)
			#plt.figure(1)
			
			clip_limit = clip_limit[np.argmax(snr)]

			#plt.imshow(it.preprocess_image(image, sigma=sigma, threshold=True, clip_limit=clip_limit))
			#plt.show()

			return clip_limit

		print(iteration, d_snr, dd_snr, (clip_limit[-1] - clip_limit[-2]) , (snr_grad[-1] - snr_grad[-2]) , abs(snr_grad[-1] - snr_grad[-2])**2, gamma)
		clip_limit.append(new_clip_limit)

		iteration += 1


def analyse_image(current_dir, input_file_name, scale=1, sigma=None, n_clusters=10, 
				ow_metric=False, ow_network=False, snr_thresh=1.0):

	cmap = 'viridis'

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'
	image_name = input_file_name.split('/')[-1]

	fig_name = ut.check_file_name(image_name, extension='tif')
	image = it.load_image(input_file_name)
	image = rescale(image, scale)

	if not np.any([ow_metric, ow_network]) and os.path.exists(data_dir + fig_name + '.npy'):
		metrics = ut.load_npy(data_dir + fig_name)

	else:
		clip_limit = optimise_equalisation(image, sigma)
		image = it.preprocess_image(image, sigma=sigma, threshold=True, clip_limit=clip_limit)

		n_tensor = it.form_nematic_tensor(image, sigma=sigma)
		j_tensor = it.form_structure_tensor(image, sigma=sigma)
		H_tensor = it.form_hessian_tensor(image, sigma=sigma)
		pix_tube = tubeness(image, sigma=sigma)

		"Perform anisotropy analysis on each pixel"

		pix_n_anis, pix_n_angle, pix_n_energy = it.tensor_analysis(n_tensor)
		pix_j_anis, pix_j_angle, pix_j_energy = it.tensor_analysis(j_tensor)
		pix_H_anis, pix_H_angle, pix_H_energy = it.tensor_analysis(H_tensor)

		"""
		filtered_energy = np.where(pix_n_energy > threshold_otsu(pix_n_energy), 1, 0)
		noise = estimate_sigma(filtered_energy, multichannel=False, average_sigmas=True)
		noise = pix_n_energy.std()
		signal = pix_n_energy.mean()

		print(signal / noise)

		if signal / noise <= snr_thresh: raise NoiseError(signal / noise, snr_thresh)
		print(" Noise threshold accepted ({} > {})".format(signal / noise, snr_thresh))
		"""

		"Perform anisotropy analysis on whole image"

		img_anis, _ , _ = it.tensor_analysis(np.mean(n_tensor, axis=(0, 1)))
		img_H_anis, _ , _ = it.tensor_analysis(np.mean(H_tensor, axis=(0, 1)))

		"Extract fibre network"
		net = it.network_extraction(data_dir + fig_name, image, pix_n_energy, n_clusters, ow_network) 
		(label_image, sorted_areas, regions, networks) = net
	
		"Analyse fibre network"
		net_res = it.network_analysis(label_image, sorted_areas, networks, j_tensor, pix_j_anis)
		(net_area, region_anis, net_linear, net_cluster, net_degree,
		fibre_waviness, net_waviness, pix_anis, coverage, solidity) = net_res

		clustering = ut.nanmean(net_cluster, weights=net_area)
		degree = ut.nanmean(net_degree, weights=net_area)
		linearity = ut.nanmean(net_linear, weights=net_area)
		region_anis = ut.nanmean(region_anis, weights=net_area)
		solidity = ut.nanmean(solidity, weights=net_area)
		fibre_waviness = ut.nanmean(fibre_waviness, weights=net_area)
		net_waviness = ut.nanmean(net_waviness, weights=net_area)

		pix_anis = np.mean(pix_anis)

		metrics = (clustering, degree, linearity, coverage, fibre_waviness, 
					net_waviness, solidity, pix_anis, region_anis, img_anis[0])

		ut.save_npy(data_dir + fig_name, metrics)

	return metrics



def analyse_directory(input_files, ow_metric=False, ow_network=False, save_db=None):

	current_dir = os.getcwd()

	scale = 1
	sigma = 0.5

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'

	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	removed_files = []

	database_array = np.empty((0, 10), dtype=float)

	for i, input_file_name in enumerate(input_files):
		try:
			res = analyse_image(current_dir, input_file_name, scale=scale, 
							sigma=sigma, ow_metric=ow_metric, ow_network=ow_network)

			database_array = np.concatenate((database_array, np.expand_dims(res, axis=0)))

			print(' Network Clustering = {:>6.4f}'.format(database_array[-1][0]))
			print(' Network Degree = {:>6.4f}'.format(database_array[-1][1]))
			print(' Network Linearity = {:>6.4f}'.format(database_array[-1][2]))
			print(' Network Coverage = {:>6.4f}'.format(database_array[-1][3]))
			print(' Network Solidity = {:>6.4f}'.format(database_array[-1][4]))
			print(' Network Waviness = {:>6.4f}'.format(database_array[-1][5]))
			print(' Av. Fibre Waviness = {:>6.4f}'.format(database_array[-1][6]))
			
			print(' Average Pixel anistoropy = {:>6.4f}'.format(database_array[-1][7]))
			print(' Average Region Anistoropy = {:>6.4f}'.format(database_array[-1][8]))
			print(' Total Image anistoropy = {:>6.4f}\n'.format(database_array[-1][9]))

		except NoiseError as err:
			print(err.message)
			removed_files.append(input_file_name)

	for file_name in removed_files: input_files.remove(file_name)

	dataframe = pd.DataFrame(data=database_array, columns=['Clustering', 'Degree', 'Linearity', 'Coverage', 
								'Fibre Waviness', 'Network Waviness', 'Solidity', 'Pixel Anis', 
								'Region Anis', 'Image Anis'], index = input_files)

	if save_db != None: dataframe.to_pickle(data_dir + '{}.pkl'.format(save_db))
