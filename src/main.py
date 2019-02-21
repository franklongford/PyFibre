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

from skimage.measure import shannon_entropy
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

	columns = ['SDI', 'Entropy', 'Anisotropy', 'Pixel Anisotropy',
				'Size', 'Linearity', 'Eccentricity', 'Density', 'Coverage',
				'Contrast', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Energy',
				'Network Waviness', 'Network Degree', 'Network Eigenvalue',
				'Network Connectivity', 'Network Local Efficiency', 'Network Clustering',
				'Segment Hu Moment 1', 'Segment Hu Moment 2', 'SegmentHole Hu Moment 3',
				'Segment Hu Moment 4', 'Segment Hu Moment 5', 'Segment Hu Moment 6',
				'Segment Hu Moment 7']

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
		image_shg, image_pl = load_image(input_file_name)
		"Pre-process image to remove noise"
		image_shg = clip_intensities(image_shg, p_intensity=p_intensity)
		image_pl = clip_intensities(image_pl, p_intensity=p_intensity)

		print("Performing Global Image analysis")

		"Perform fourier analysis to obtain spectrum and sdi metric"
		angles, fourier_spec, global_sdi = an.fourier_transform_analysis(image_shg)

		"Form nematic and structure tensors for each pixel"
		n_tensor = form_nematic_tensor(image_shg, sigma=sigma)
		j_tensor = form_structure_tensor(image_shg, sigma=sigma)

		"Perform anisotropy analysis on each pixel"
		pix_n_anis, pix_n_angle, pix_n_energy = an.tensor_analysis(n_tensor)
		pix_j_anis, pix_j_angle, pix_j_energy = an.tensor_analysis(j_tensor)

		"Perform anisotropy analysis on whole image"
		img_n_anis, _ , _ = an.tensor_analysis(np.mean(n_tensor, axis=(0, 1)))
		img_j_anis, _ , _ = an.tensor_analysis(np.mean(j_tensor, axis=(0, 1)))

		global_entropy = shannon_entropy(image_shg)
		global_anis = img_n_anis[0]
		global_pix_anis = np.mean(pix_n_anis)

		print("Segmenting Image using FIRE")

		"Extract fibre network"
		net = seg.network_extraction(image_shg, data_dir + image_name, p_denoise=p_denoise,
						ow_network=ow_network, threads=threads) 
		(segmented_image, networks, areas, segments) = net
	
		global_size = np.count_nonzero(segmented_image)
		global_coverage = global_size / segmented_image.size

		print("Performing Segmented Image analysis")

		"Analyse fibre network"
		net_res = seg.network_analysis(image_shg, image_pl, networks, segments, n_tensor, pix_n_anis)
		(segment_sdi, segment_entropy, segment_anis, segment_pix_anis, segment_size,
		segment_linear, segment_eccent, segment_density, local_coverage,
		segment_contrast, segment_homo, segment_dissim, segment_corr, segment_energy, segment_hu,
		network_waviness, network_degree, network_eigen, 
		network_connect, network_loc_eff, network_cluster) = net_res

		"Calculate area-weighted metrics for segmented image"
		av_segment_linear = ut.nanmean(segment_linear, weights=areas)
		av_segment_eccent = ut.nanmean(segment_eccent, weights=areas)
		av_segment_density = ut.nanmean(segment_density, weights=areas)
		av_segment_contrast = ut.nanmean(segment_contrast, weights=areas)
		av_segment_homo = ut.nanmean(segment_homo, weights=areas)
		av_segment_dissim = ut.nanmean(segment_dissim, weights=areas)
		av_segment_corr = ut.nanmean(segment_corr, weights=areas)
		av_segment_energy = ut.nanmean(segment_energy, weights=areas)

		av_network_waviness = ut.nanmean(network_waviness, weights=areas)
		av_network_degree = ut.nanmean(network_degree, weights=areas)
		av_network_eigen = ut.nanmean(network_eigen, weights=areas)
		av_network_connect = ut.nanmean(network_connect, weights=areas)
		av_network_loc_eff = ut.nanmean(network_loc_eff, weights=areas)
		av_network_cluster = ut.nanmean(network_cluster, weights=areas)

		av_segment_hu = np.zeros(7)
		for j in range(7): av_segment_hu[j] = ut.nanmean(segment_hu[:,j], weights=areas)

		global_metrics = np.array([
					global_sdi, global_entropy, global_anis, global_pix_anis, global_size,
					av_segment_linear, av_segment_eccent, av_segment_density, global_coverage,
					av_segment_contrast, av_segment_homo, av_segment_dissim, av_segment_corr, av_segment_energy,
					av_network_waviness, av_network_degree, av_network_eigen, 
					av_network_connect, av_network_loc_eff, av_network_cluster, av_segment_hu])
		global_metrics = np.concatenate((global_metrics, av_hole_hu))
		global_metrics = np.expand_dims(global_metrics, axis=0)

		segment_metrics = np.stack([
					segment_sdi, segment_entropy, segment_anis, segment_pix_anis, segment_size,
					segment_linear, segment_eccent, segment_density, local_coverage,
					segment_contrast, segment_homo, segment_dissim, segment_corr, segment_energy,
					network_waviness, network_degree, network_eigen, 
					network_connect, network_loc_eff, network_cluster], axis=-1)

		segment_metrics = np.concatenate((segment_metrics, segment_hu), axis=-1)

		titles = [input_file_name]
		for i in range(len(segments)): titles += [input_file_name + "_{}".format(i)]

		dataframe = pd.DataFrame(data=np.concatenate((global_metrics, segment_metrics), axis=0), 
								columns=columns, index=titles)
 
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
		#elif (file_name.find('AVG') == -1): removed_files.append(file_name)

	for key in args.key.split(','):
		for file_name in input_files:
			if (file_name.find(key) == -1) and (file_name not in removed_files): 
				removed_files.append(file_name)
		
	for file_name in removed_files: input_files.remove(file_name)

	print(input_files)

	removed_files = []
	segment_database = pd.DataFrame()
	global_database = pd.DataFrame()
   
	for i, input_file_name in enumerate(input_files):

		image_path = '/'.join(input_file_name.split('/')[:-1])

		data = analyse_image(input_file_name, image_path, sigma=args.sigma, 
			ow_metric=args.ow_metric, ow_network=args.ow_network, threads=args.threads)

		global_database = pd.concat([global_database, data.iloc[:1]])
		segment_database = pd.concat([segment_database, data.iloc[1:]])

		for i, title in enumerate(global_database.columns): 
			print(' {} = {:>6.4f}'.format(title, global_database.loc[input_file_name][title]))

	for file_name in removed_files: input_files.remove(file_name)

	if args.save_db != None: 
		global_database.to_pickle('{}_global.pkl'.format(args.save_db))
		segment_database.to_excel('{}_segment.xls'.format(args.save_db))
		global_database.to_pickle('{}_global.pkl'.format(args.save_db))
		segment_database.to_excel('{}_segment.xls'.format(args.save_db))

		
