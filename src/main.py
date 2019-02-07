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

from multiprocessing import Pool, Process, JoinableQueue, Queue, current_process

import utilities as ut
from utilities import NoiseError
import image_tools as it


def analyse_image(input_file_name, working_dir=None, scale=1, 
				p_intensity=(1, 98), p_denoise=(12, 35), sigma=0.5, 
				ow_metric=False, ow_network=False, threads=8):
	"""
	Analyse imput image by calculating metrics and sgenmenting via FIRE algorithm

	Parameters
	----------

	input_file_name: str
		Full file path of image

	working_dir: str (optional)
		Working directory

	scale: float (optional)
		Unit of scale to resize image

	p_intensity: tuple (float); shape=(2,)
		Percentile range for intensity rescaling (used to remove outliers)
	
	p_denoise: tuple (float); shape=(2,)
		Parameters for non-linear means denoise algorithm (used to remove noise)

	sigma: float (optional)
		Standard deviation of Gaussian smoothing

	ow_metric: bool (optional)
		Force over-write of image metrics

	ow_network: bool (optional)
		Force over-write of image network

	threads: int (optional)
		Maximum number of threads to use for FIRE algorithm

	Returns
	-------

	metrics: array_like, shape=(11,)
		Calculated metrics for further analysis
	"""

	columns = ['Global SDI', 'Global Pixel Anisotropy', 'Global Anisotropy', 'Global Coverage',
				'Local SDI', 'Local Pixel Anisotropy', 'Local Anisotropy',
				'Linearity', 'Eccentricity', 'Density',
				'Network Waviness', 'Network Degree', 'Network Centrality',
				'Network Connectivity', 'Network Local Efficiency']

	cmap = 'viridis'
	if working_dir == None: working_dir = os.getcwd()

	data_dir = working_dir + '/data/'	
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	file_name = input_file_name.split('/')[-1]
	image_name = ut.check_file_name(file_name, extension='tif')

	if not np.any([ow_metric, ow_network]) and os.path.exists(data_dir + image_name + '_metric.pkl'):
		dataframe = pd.read_pickle('{}_metric.pkl'.format(data_dir + image_name))

	else:
		print(f"Loading image {data_dir + image_name}")
		"Load and preprocess image"
		image = it.load_image(input_file_name)
		"Pre-process image to remove noise"
		image = it.preprocess_image(image, scale=scale, p_intensity=p_intensity,
										p_denoise=p_denoise)

		"Perform fourier analysis to obtain spectrum and sdi metric"
		print("Performing Fourier analysis")
		angles, fourier_spec, global_sdi = it.fourier_transform_analysis(image)

		"Form nematic and structure tensors for each pixel"
		n_tensor = it.form_nematic_tensor(image, sigma=sigma)
		j_tensor = it.form_structure_tensor(image, sigma=sigma)

		"Perform anisotropy analysis on each pixel"
		pix_n_anis, pix_n_angle, pix_n_energy = it.tensor_analysis(n_tensor)
		pix_j_anis, pix_j_angle, pix_j_energy = it.tensor_analysis(j_tensor)

		"Perform anisotropy analysis on whole image"
		img_n_anis, _ , _ = it.tensor_analysis(np.mean(n_tensor, axis=(0, 1)))
		img_j_anis, _ , _ = it.tensor_analysis(np.mean(j_tensor, axis=(0, 1)))

		global_anis = img_n_anis[0]
		global_pix_anis = np.mean(pix_n_anis)

		"Extract fibre network"
		net = it.network_extraction(image, data_dir + image_name, ow_network=ow_network, threads=threads) 
		(segmented_image, networks, areas, regions, segments) = net
	
		"Analyse fibre network"
		net_res = it.network_analysis(image, segmented_image, networks, regions, segments, n_tensor, pix_n_anis)
		(global_coverage, region_sdi, region_anis, region_pix_anis, 
		segment_linear, segment_eccent, segment_density,
		network_waviness, network_degree, network_central, 
		network_connect, network_loc_eff) = net_res

		"Calculate area-weighted metrics for segmented image"
		local_sdi = ut.nanmean(region_sdi, weights=areas)
		local_anis = ut.nanmean(region_anis, weights=areas)
		local_pix_anis = ut.nanmean(region_pix_anis, weights=areas)

		segment_linear = ut.nanmean(segment_linear, weights=areas)
		segment_eccent = ut.nanmean(segment_eccent, weights=areas)
		segment_density = ut.nanmean(segment_density, weights=areas)

		network_waviness = ut.nanmean(network_waviness, weights=areas)
		network_degree = ut.nanmean(network_degree, weights=areas)
		network_central = ut.nanmean(network_central, weights=areas)
		network_connect = ut.nanmean(network_connect, weights=areas)
		network_loc_eff = ut.nanmean(network_loc_eff, weights=areas)

		metrics = np.array([global_sdi, global_anis, global_pix_anis, global_coverage, 
					local_sdi, local_anis, local_pix_anis, 
					segment_linear, segment_eccent, segment_density,
					network_waviness, network_degree, network_central, 
					network_connect, network_loc_eff])

		dataframe = pd.DataFrame(data=[metrics], columns=columns, 
				index = [input_file_name])
 
		dataframe.to_pickle('{}_metric.pkl'.format(data_dir + image_name))


	return dataframe


if __name__ == '__main__':

	current_dir = os.getcwd()
	dir_path = os.path.dirname(os.path.realpath(__file__))

	print(ut.logo())

	parser = argparse.ArgumentParser(description='Image analysis of fibourous tissue samples')
	parser.add_argument('--name', nargs='?', help='Tif file names to load', default="")
	parser.add_argument('--dir', nargs='?', help='Directories to load tif files', default="")
	parser.add_argument('--key', nargs='?', help='Keywords to filter file names', default="")
	parser.add_argument('--sigma', type=float, nargs='?', help='Gaussian smoothing standard deviation', default=0.5)
	parser.add_argument('--ow_metric', action='store_true', help='Toggles overwrite analytic metrics')
	parser.add_argument('--ow_network', action='store_true', help='Toggles overwrite network extraction')
	parser.add_argument('--save_db', nargs='?', help='Output database filename', default=None)
	parser.add_argument('--threads', type=int, nargs='?', help='Number of threads per processor', default=8)
	args = parser.parse_args()

	input_files = []

	for file_name in args.name.split(','):
		if (file_name.find('/') == -1): file_name = current_dir + '/' + file_name
		input_files.append(file_name)

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

	print(input_files)

	removed_files = []
	database = pd.DataFrame()
   
	for i, input_file_name in enumerate(input_files):

		image_path = '/'.join(input_file_name.split('/')[:-1])

		try:
			data = analyse_image(input_file_name, image_path, sigma=args.sigma, 
				ow_metric=args.ow_metric, ow_network=args.ow_network, threads=args.threads)

			database = pd.concat([database, data])

			print(input_file_name)
			for i, title in enumerate(database.columns): 
				print(' {} = {:>6.4f}'.format(title, database.loc[input_file_name][title]))

		except NoiseError as err:
			print(err.message)
			removed_files.append(input_file_name)

	for file_name in removed_files: input_files.remove(file_name)

	if args.save_db != None: 
		database.to_pickle('{}.pkl'.format(args.save_db))
		database.to_excel('{}.xls'.format(args.save_db))


		
