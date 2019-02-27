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

from scipy.ndimage.filters import gaussian_filter

from skimage.measure import shannon_entropy, regionprops
from multiprocessing import Pool, Process, JoinableQueue, Queue, current_process

import utilities as ut
from preprocessing import import_image, clip_intensities
import analysis as an
import segmentation as seg
from filters import form_nematic_tensor, form_structure_tensor


def analyse_image(input_file_name, working_dir=None, scale=1, 
				p_intensity=(1, 99), p_denoise=(2, 25), sigma=0.5, alpha=0.5,
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

	global_columns = ['Fourier SDI', 'Angle SDI', 'Entropy', 'Anisotropy', 'Pixel Anisotropy', 'Area',
			'Linearity', 'Eccentricity', 'Density', 'Coverage',
			'Contrast', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Energy', 'No. Fibres',
			'No. Cells',
			'Hu Moment 1', 'Hu Moment 2', 'Hu Moment 3', 'Hu Moment 4', 
			'Hu Moment 5', 'Hu Moment 6', 'Hu Moment 7']

	fibre_columns = ['Fourier SDI', 'Angle SDI', 'Entropy', 'Anisotropy', 'Pixel Anisotropy',
				'Area', 'Linearity', 'Eccentricity', 'Density', 'Coverage',
				'Contrast', 'Homogeneity', 'Dissimilarity', 'Correlation', 'Energy',
				'Network Waviness', 'Network Degree', 'Network Eigenvalue',
				'Network Connectivity', 'Network Local Efficiency', 'Network Clustering',
				'Hu Moment 1', 'Hu Moment 2', 'Hu Moment 3',
				'Hu Moment 4', 'Hu Moment 5', 'Hu Moment 6',
				'Hu Moment 7']

	cell_columns = ['Area', 'Contrast', 'Homogeneity', 'Dissimilarity', 'Correlation', 
			'Energy','Hu Moment 1', 'Hu Moment 2', 'Hu Moment 3', 'Hu Moment 4', 
			'Hu Moment 5', 'Hu Moment 6','Hu Moment 7']

	cmap = 'viridis'
	if working_dir == None: working_dir = os.getcwd()

	data_dir = working_dir + '/data/'	
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	file_name = input_file_name.split('/')[-1]
	image_name = ut.check_file_name(file_name, extension='tif')

	if not np.any([ow_metric, ow_network]):
		try: 
			global_dataframe = pd.read_pickle('{}_global_metric.pkl'.format(data_dir + image_name))
			fibre_dataframe = pd.read_pickle('{}_fibre_metric.pkl'.format(data_dir + image_name))
			cell_dataframe = pd.read_pickle('{}_cell_metric.pkl'.format(data_dir + image_name))
		except IOError:
			print("Cannot load metrics for {}".format(image_name))
			ow_metric = True

	if ow_metric:
		print(f"Loading image {data_dir + image_name}")
		"Load and preprocess image"
		image_shg, image_pl = import_image(input_file_name)
		"Pre-process image to remove noise"
		image_shg = clip_intensities(image_shg, p_intensity=p_intensity)
		image_pl = clip_intensities(image_pl, p_intensity=p_intensity)

		filename = '{}'.format(data_dir + image_name)

		"Form nematic and structure tensors for each pixel"
		n_tensor = form_nematic_tensor(image_shg, sigma=sigma)
		j_tensor = form_structure_tensor(image_shg, sigma=sigma)

		"Perform anisotropy analysis on each pixel"
		pix_n_anis, pix_n_angle, pix_n_energy = an.tensor_analysis(n_tensor)
		pix_j_anis, pix_j_angle, pix_j_energy = an.tensor_analysis(j_tensor)

		print("Performing cell segment analysis")
		
		cells, hole_labels = seg.hole_extraction(image_shg, image_pl)

		hole_filter = np.where(hole_labels, alpha, 1)
		hole_filter = gaussian_filter(hole_filter, sigma=0.5)

		(cell_areas, cell_contrast, cell_homo, cell_dissim, 
		cell_corr, cell_energy, cell_hu) = seg.hole_analysis(image_pl, cells)

		filenames = pd.Series(['{}_cell_segment.pkl'.format(filename)] * len(cells), name='File')		
		cell_id = pd.Series(np.arange(len(cells)), name='ID')

		cell_metrics = np.stack([cell_areas, cell_contrast, cell_homo, 
					cell_dissim, cell_corr, cell_energy], axis=-1)
		cell_metrics = np.concatenate((cell_metrics, cell_hu), axis=-1)

		cell_dataframe = pd.DataFrame(data=cell_metrics, columns=cell_columns)
		cell_dataframe = pd.concat((filenames, cell_id, cell_dataframe), axis=1)
		cell_dataframe.to_pickle('{}_cell_metric.pkl'.format(filename))
		ut.save_region(cells, '{}_cell_segment'.format(filename))

		"Extract fibre network"
		net = seg.network_extraction(image_shg * hole_filter, data_dir + image_name,
						sigma=sigma, p_denoise=p_denoise,
						ow_network=ow_network, threads=threads) 
		(segmented_image, networks, networks_red, fibres) = net
	
		print("Performing fibre segment analysis")

		"Analyse fibre network"
		net_res = seg.network_analysis(image_shg, image_pl, networks, networks_red, fibres, 
										j_tensor, pix_j_anis, pix_j_angle)
		(fibre_fourier_sdi, fibre_angle_sdi, fibre_entropy, fibre_anis, fibre_pix_anis, 
		fibre_areas, fibre_linear, fibre_eccent, fibre_density, fibre_coverage,
		fibre_contrast, fibre_homo, fibre_dissim, fibre_corr, fibre_energy, fibre_hu,
		network_waviness, network_degree, network_eigen, 
		network_connect, network_loc_eff, network_cluster) = net_res

		filenames = pd.Series(['{}_fibre_segment.pkl'.format(filename)] * len(fibres), name='File')
		fibre_id = pd.Series(np.arange(len(fibres)), name='ID')

		fibre_metrics = np.stack([fibre_fourier_sdi, fibre_angle_sdi, fibre_entropy, fibre_anis, 
					fibre_pix_anis, fibre_areas, fibre_linear, fibre_eccent, fibre_density,
					fibre_coverage, fibre_contrast, fibre_homo, fibre_dissim, fibre_corr,
					fibre_energy, network_waviness, network_degree, network_eigen, 
					network_connect, network_loc_eff, network_cluster], axis=-1)
		fibre_metrics = np.concatenate((fibre_metrics, fibre_hu), axis=-1)

		fibre_dataframe = pd.DataFrame(data=fibre_metrics, columns=fibre_columns)
		fibre_dataframe = pd.concat((filenames, fibre_id, fibre_dataframe), axis=1)
		fibre_dataframe.to_pickle('{}_fibre_metric.pkl'.format(filename))
		ut.save_region(fibres, '{}_fibre_segment'.format(filename))

		print("Performing Global Image analysis")

		global_binary = np.where(hole_labels, 0, 1)
		global_segment = regionprops(global_binary)[0]

		metrics = seg.segment_analysis(image_shg, image_pl, global_segment, j_tensor, 
						pix_j_anis, pix_j_angle)

		(global_fourier_sdi, global_angle_sdi, global_entropy, global_anis, global_pix_anis, 
		global_area, global_linear, global_eccent, global_density, 
		global_coverage, global_contrast, global_homo, global_dissim, 
		global_corr, global_energy, global_hu) = metrics

		filenames = pd.Series('{}_global_segment.pkl'.format(filename), name='File')

		global_metrics = np.stack([
					global_fourier_sdi, global_angle_sdi, global_entropy, global_anis, 
					global_pix_anis, global_area, global_linear, global_eccent, 
					global_density, global_coverage, global_contrast, global_homo, 
					global_dissim, global_corr, global_energy, (len(fibres)),
					(len(cells))], axis=0)
		global_metrics = np.concatenate((global_metrics, global_hu), axis=0)
		global_metrics = np.expand_dims(global_metrics, 0)

		global_dataframe = pd.DataFrame(data=global_metrics, columns=global_columns)
		global_dataframe = pd.concat((filenames, global_dataframe), axis=1)
		global_dataframe.to_pickle('{}_global_metric.pkl'.format(filename))
		ut.save_region(global_segment, '{}_global_segment'.format(filename))


	return global_dataframe, fibre_dataframe, cell_dataframe


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

		
