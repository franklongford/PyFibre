"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE 

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 18/02/2019
"""

import sys, os
import argparse

import numpy as np
import pandas as pd

from skimage.measure import shannon_entropy, regionprops
from multiprocessing import Pool, Process, JoinableQueue, Queue, current_process

import utilities as ut
from preprocessing import load_image, clip_intensities
import analysis as an
import segmentation as seg
from filters import form_nematic_tensor, form_structure_tensor


def analyse_image(input_file_name, working_dir=None, scale=1, 
				p_intensity=(1, 98), p_denoise=(2, 35), sigma=0.5, 
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

	columns_global = ['SDI', 'Entropy', 'Anisotropy', 'Pixel Anisotropy',
			'Linearity', 'Eccentricity', 'Density', 'Coverage',
			'Contrast', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Energy']

	columns_segment = ['SDI', 'Entropy', 'Anisotropy', 'Pixel Anisotropy',
				'Area', 'Linearity', 'Eccentricity', 'Density', 'Coverage',
				'Contrast', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Energy',
				'Network Waviness', 'Network Degree', 'Network Eigenvalue',
				'Network Connectivity', 'Network Local Efficiency', 'Network Clustering',
				'Hu Moment 1', 'Hu Moment 2', 'Hu Moment 3',
				'Hu Moment 4', 'Hu Moment 5', 'Hu Moment 6',
				'Hu Moment 7']

	columns_holes = ['Area', 'Contrast', 'Homogeneity', 'Dissimilarity', 'Correlation', 
			'Energy','Hu Moment 1', 'Hu Moment 2', 'Hu Moment 3', 'Hu Moment 4', 
			'Hu Moment 5', 'Hu Moment 6','Hu Moment 7']

	cmap = 'viridis'
	if working_dir == None: working_dir = os.getcwd()

	data_dir = working_dir + '/data/'	
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	file_name = input_file_name.split('/')[-1]
	image_name = ut.check_file_name(file_name, extension='tif')

	if not np.any([ow_metric, ow_network]) and os.path.exists(data_dir + image_name + '_metric.pkl'):
		try: 
			dataframe_global = pd.read_pickle('{}_global_metric.pkl'.format(data_dir + image_name))
			dataframe_segment = pd.read_pickle('{}_segment_metric.pkl'.format(data_dir + image_name))
			dataframe_hole = pd.read_pickle('{}_hole_metric.pkl'.format(data_dir + image_name))
		except IOError: ow_metric = True
	else:
		print(f"Loading image {data_dir + image_name}")
		"Load and preprocess image"
		image_shg, image_pl = load_image(input_file_name)
		"Pre-process image to remove noise"
		image_shg = clip_intensities(image_shg, p_intensity=p_intensity)
		image_pl = clip_intensities(image_pl, p_intensity=p_intensity)
		image_combined = np.sqrt(image_shg * image_pl)

		print("Performing Global Image analysis")

		"Perform fourier analysis to obtain spectrum and sdi metric"
		angles, fourier_spec, global_sdi = an.fourier_transform_analysis(image_shg)

		"Form nematic and structure tensors for each pixel"
		n_tensor = form_nematic_tensor(image_shg, sigma=sigma)
		j_tensor = form_structure_tensor(image_shg, sigma=sigma)

		"Perform anisotropy analysis on each pixel"
		pix_n_anis, pix_n_angle, pix_n_energy = an.tensor_analysis(n_tensor)
		pix_j_anis, pix_j_angle, pix_j_energy = an.tensor_analysis(j_tensor)

		holes, hole_labels = seg.hole_extraction(image_combined)

		global_binary = np.where(hole_labels, 0, 1)
		global_segment = regionprops(global_binary)[0]

		metrics = seg.segment_analysis(image_shg, image_pl, global_segment, j_tensor, pix_j_anis)

		(global_sdi, global_entropy, global_anis, global_pix_anis, 
		global_area, global_linear, global_eccent, global_density, 
		global_coverage, global_contrast, global_homo, global_dissim, 
		global_corr, global_energy, global_hu) = metrics

		global_metrics = np.array([
					global_sdi, global_entropy, global_anis, global_pix_anis, 
					global_linear, global_eccent, global_density, 
					global_coverage, global_contrast, global_homo, global_dissim, 
					global_corr, global_energy])
		global_metrics = np.expand_dims(global_metrics, axis=0)

		dataframe_global = pd.DataFrame(data=global_metrics, columns=columns_global, index=[input_file_name])
		dataframe_global.to_pickle('{}_global_metric.pkl'.format(data_dir + image_name))

		print("Performing Hole Image analysis")

		(hole_areas, hole_contrast, hole_homo, hole_dissim, 
		hole_corr, hole_energy, hole_hu) = seg.hole_analysis(image_pl, holes)

		av_hole_hu = np.zeros(7)
		for j in range(7): av_hole_hu[j] = ut.nanmean(hole_hu[:,j], weights=hole_areas)

		hole_metrics = np.stack([hole_areas, hole_contrast, hole_homo, hole_dissim, 
					 hole_corr, hole_energy], axis=-1)
		hole_metrics = np.concatenate((hole_metrics, hole_hu), axis=-1)

		titles = []
		for i in range(len(holes)): titles += [input_file_name + "_hole_{}".format(i)]

		dataframe_hole = pd.DataFrame(data=hole_metrics, columns=columns_holes, index=titles)
		dataframe_hole.to_pickle('{}_hole_metric.pkl'.format(data_dir + image_name))
		ut.save_region(holes, '{}_holes'.format(data_dir + image_name))

		"Extract fibre network"
		net = seg.network_extraction(image_shg, data_dir + image_name, p_denoise=p_denoise,
						ow_network=ow_network, threads=threads) 
		(segmented_image, networks, segments) = net
	
		print("Performing Segmented Image analysis")

		"Analyse fibre network"
		net_res = seg.network_analysis(image_shg, image_pl, networks, segments, j_tensor, pix_j_anis)
		(segment_sdi, segment_entropy, segment_anis, segment_pix_anis, segment_areas,
		segment_linear, segment_eccent, segment_density, local_coverage,
		segment_contrast, segment_homo, segment_dissim, segment_corr, segment_energy, segment_hu,
		network_waviness, network_degree, network_eigen, 
		network_connect, network_loc_eff, network_cluster) = net_res

		av_segment_hu = np.zeros(7)
		for j in range(7): av_segment_hu[j] = ut.nanmean(segment_hu[:,j], weights=segment_areas)

		segment_metrics = np.stack([
					segment_sdi, segment_entropy, segment_anis, segment_pix_anis, segment_areas,
					segment_linear, segment_eccent, segment_density, local_coverage,
					segment_contrast, segment_homo, segment_dissim, segment_corr, segment_energy,
					network_waviness, network_degree, network_eigen, 
					network_connect, network_loc_eff, network_cluster], axis=-1)
		segment_metrics = np.concatenate((segment_metrics, segment_hu), axis=-1)

		titles = []
		for i in range(len(segments)): titles += [input_file_name + "_segment_{}".format(i)]

		dataframe_segment = pd.DataFrame(data=segment_metrics, columns=columns_segment, index=titles)
		dataframe_segment.to_pickle('{}_segment_metric.pkl'.format(data_dir + image_name))
		ut.save_region(segments, '{}_segments'.format(data_dir + image_name))

	return dataframe_global, dataframe_segment, dataframe_hole


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
		#elif (file_name.find('AVG') == -1): removed_files.append(file_name)

	for key in args.key.split(','):
		for file_name in input_files:
			if (file_name.find(key) == -1) and (file_name not in removed_files): 
				removed_files.append(file_name)
		
	for file_name in removed_files: input_files.remove(file_name)

	removed_files = []
	hole_database = pd.DataFrame()
	segment_database = pd.DataFrame()
	global_database = pd.DataFrame()
   
	for i, input_file_name in enumerate(input_files):

		image_path = '/'.join(input_file_name.split('/')[:-1])

		data_global, data_segment, data_hole = analyse_image(input_file_name, image_path, sigma=args.sigma, 
			ow_metric=args.ow_metric, ow_network=args.ow_network, threads=args.threads)

		global_database = pd.concat([global_database, data_global])
		segment_database = pd.concat([segment_database, data_segment])
		hole_database = pd.concat([hole_database, data_hole])

		print(input_file_name.split('/')[-1])
		print("Global Image Analysis Metrics:")
		print(data_global.iloc[0])

	for file_name in removed_files: input_files.remove(file_name)

	if args.save_db != None: 
		global_database.to_pickle('{}.pkl'.format(args.save_db))
		global_database.to_excel('{}.xls'.format(args.save_db))
		
		segment_database.to_pickle('{}_segment.pkl'.format(args.save_db))
		segment_database.to_excel('{}_segment.xls'.format(args.save_db))

		hole_database.to_pickle('{}_hole.pkl'.format(args.save_db))
		hole_database.to_excel('{}_hole.xls'.format(args.save_db))

		
