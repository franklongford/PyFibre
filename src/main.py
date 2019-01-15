"""
ImageCol: Collagen Image Analysis Program
MAIN ROUTINE 

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 16/08/2018
"""

import sys, os
import argparse

import numpy as np
import pandas as pd

from skimage.transform import rescale

import utilities as ut
from utilities import NoiseError
import image_tools as it


def analyse_image(current_dir, input_file_name, scale=1, sigma=None, n_clusters=10, 
				ow_metric=False, ow_network=False, noise_thresh=0.15, threads=8):

	cmap = 'viridis'

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'
	
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	image_name = input_file_name.split('/')[-1]

	fig_name = ut.check_file_name(image_name, extension='tif')
	image = it.load_image(input_file_name)
	image = rescale(image, scale)

	if not np.any([ow_metric, ow_network]) and os.path.exists(data_dir + fig_name + '.npy'):
		metrics = ut.load_npy(data_dir + fig_name)

	else:
		n_tensor = it.form_nematic_tensor(image, sigma=sigma)
		j_tensor = it.form_structure_tensor(image, sigma=sigma)
		H_tensor = it.form_hessian_tensor(image, sigma=sigma)

		"Perform anisotropy analysis on each pixel"

		pix_n_anis, pix_n_angle, pix_n_energy = it.tensor_analysis(n_tensor)
		pix_j_anis, pix_j_angle, pix_j_energy = it.tensor_analysis(j_tensor)
		pix_H_anis, pix_H_angle, pix_H_energy = it.tensor_analysis(H_tensor)

		"Perform anisotropy analysis on whole image"

		img_anis, _ , _ = it.tensor_analysis(np.mean(n_tensor, axis=(0, 1)))
		img_H_anis, _ , _ = it.tensor_analysis(np.mean(H_tensor, axis=(0, 1)))

		"Extract fibre network"
		"""
		clip_limit, weight, snr = it.optimise_equalisation(image)
		print(f"clip_limit = {clip_limit}, sigma = {weight}, signal / noise = {snr}")
		if snr <= snr_thresh: raise NoiseError(snr, snr_thresh)
		print(" Noise threshold accepted ({} > {})".format(snr, snr_thresh))
		#"""
		pre_image, noise = it.preprocess_image(image, threshold=True, sigma=0.5, interval=0.9, clip_limit=0.01)
		#if noise >= noise_thresh: raise NoiseError(noise, noise_thresh)
		#print(" Noise threshold accepted ({} > {})".format(noise, noise_thresh))

		net = it.network_extraction(data_dir + fig_name, pre_image, ow_network, threads) 
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



def analyse_directory(input_files, ow_metric=False, ow_network=False, save_db=None, threads=8):

	scale = 1
	sigma = 0.5

	removed_files = []

	database_array = np.empty((0, 10), dtype=float)

	for i, input_file_name in enumerate(input_files):
		try:
			res = analyse_image(current_dir, input_file_name, scale=scale, 
					  sigma=sigma, ow_metric=ow_metric, ow_network=ow_network, 
					  threads=threads)

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


if __name__ == '__main__':

	current_dir = os.getcwd()
	dir_path = os.path.dirname(os.path.realpath(__file__))

	print(ut.logo())

	parser = argparse.ArgumentParser(description='Image analysis of fibourous tissue samples')
	parser.add_argument('--name', nargs='?', help='Tif file names to load', default="")
	parser.add_argument('--dir', nargs='?', help='Directories to load tif files', default="")
	parser.add_argument('--key', nargs='?', help='Keywords to filter file names', default="")
	parser.add_argument('--ow_metric', action='store_true', help='Toggles overwrite analytic metrics')
	parser.add_argument('--ow_network', action='store_true', help='Toggles overwrite network extraction')
	parser.add_argument('--save_db', nargs='?', help='Output database filename', default=None)
	parser.add_argument('--threads', type=int, nargs='?', help='Number of threads per processor', default=8)
	args = parser.parse_args()

	print(args)

	input_files = args.name.split(',')

	if len(args.dir) != 0:
		for directory in args.dir.split(','): 
			for file_name in os.listdir(directory): input_files += [directory + '/' + file_name]

	removed_files = []

	for file_name in input_files:
		if not (file_name.endswith('.tif')): removed_files.append(file_name)
		elif (file_name.find('display') != -1): removed_files.append(file_name)
		elif (file_name.find('AVG') == -1): removed_files.append(file_name)

	for key in args.key.split(','):
		for file_name in input_files:
			if (file_name.find(key) == -1) and (file_name not in removed_files): 
				removed_files.append(file_name)
		
	for file_name in removed_files: input_files.remove(file_name)

	analyse_directory(input_files, args.ow_metric, args.ow_network, args.save_db, args.threads)

		
