"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE 

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 18/02/2019
"""

import sys, os, time
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from scipy.ndimage.filters import gaussian_filter

from skimage.measure import shannon_entropy, regionprops
from skimage.exposure import equalize_adapthist

from multiprocessing import Pool, Process, JoinableQueue, Queue, current_process

import utilities as ut
from preprocessing import load_shg_pl, clip_intensities
import analysis as an
import segmentation as seg
from filters import form_nematic_tensor, form_structure_tensor
from figures import create_figure, create_tensor_image, create_region_image, create_network_image


def analyse_image(input_file_names, prefix, working_dir=None, scale=1, 
				p_intensity=(1, 99), p_denoise=(2, 25), sigma=0.5, alpha=0.5,
				ow_metric=False, ow_segment=False, ow_network=False, ow_figure=False,
				threads=8):
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

	fibre_columns = ['SHG Angle SDI', 'SHG Anisotropy', 'SHG Pixel Anisotropy',
					'Fibre Area', 'Fibre Linearity', 'Fibre Eccentricity', 'Fibre Density',
					'Fibre Coverage', 'SHG Intensity Mean', 'SHG Intensity STD', 
					'SHG Intensity Entropy', 'Fibre GLCM Contrast', 'Fibre GLCM Homogeneity',
					'Fibre GLCM Dissimilarity', 'Fibre GLCM Correlation',
					'Fibre GLCM Energy', 'Fibre GLCM IDM', 'Fibre GLCM Variance', 
					'Fibre GLCM Cluster', 'Fibre GLCM Entropy',
					'Network Degree', 'Network Eigenvalue', 'Network Connectivity',
					'Fibre Waviness', 'Fibre Lengths', 'Fibre Cross-Link Density',
					'Fibre Hu Moment 1', 'Fibre Hu Moment 2', 'Fibre Hu Moment 3',
					'Fibre Hu Moment 4', 'Fibre Hu Moment 5', 'Fibre Hu Moment 6',
					'Fibre Hu Moment 7']

	cell_columns = ['PL Angle SDI', 'PL Anisotropy', 'PL Pixel Anisotropy',
					'Cell Area', 'PL Intensity Mean', 'PL Intensity STD', 'PL Intensity Entropy',
					'Cell GLCM Contrast', 'Cell GLCM Homogeneity',
					'Cell GLCM Dissimilarity', 'Cell GLCM Correlation',
					'Cell GLCM Energy', 'Cell GLCM IDM', 'Cell GLCM Variance', 
					'Cell GLCM Cluster', 'Cell GLCM Entropy', 
					'Cell Linearity', 'Cell Eccentricity', 'Cell Density', 'Cell Coverage',
					'Cell Hu Moment 1', 'Cell Hu Moment 2', 'Cell Hu Moment 3', 
					'Cell Hu Moment 4', 'Cell Hu Moment 5', 'Cell Hu Moment 6','Cell Hu Moment 7']

	muscle_columns = ['Muscle GLCM Contrast', 'Muscle GLCM Homogeneity',
					'Muscle GLCM Dissimilarity', 'Muscle GLCM Correlation',
					'Muscle GLCM Energy', 'Muscle GLCM IDM', 'Muscle GLCM Variance', 
					'Muscle GLCM Cluster', 'Muscle GLCM Entropy']

	cmap = 'viridis'
	if working_dir == None: working_dir = os.getcwd()

	data_dir = working_dir + '/data/'
	fig_dir = working_dir + '/fig/'

	if not os.path.exists(data_dir): os.mkdir(data_dir)
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)

	image_name = prefix.split('/')[-1]
	filename = '{}'.format(data_dir + image_name)

	print(f"Loading images for {prefix}")

	"Load and preprocess image"
	image_shg, image_pl, image_tran = load_shg_pl(input_file_names)
	pl_analysis = ~np.any(image_pl == None) * ~np.any(image_tran == None)

	print(pl_analysis)

	"Pre-process image to remove noise"
	image_shg = clip_intensities(image_shg, p_intensity=p_intensity)
	if pl_analysis: 
		image_pl = clip_intensities(image_pl, p_intensity=p_intensity)
		image_tran = equalize_adapthist(image_tran)
	try:
		networks = ut.load_region(data_dir + image_name + "_network")
		networks_red = ut.load_region(data_dir + image_name + "_network_reduced")
		fibres = ut.load_region(data_dir + image_name + "_fibre")
	except IOError:
		print("Cannot load networks for {}".format(image_name))
		ow_network = True

	try:
		fibre_seg = ut.load_region(data_dir + image_name + "_fibre_segment")
		if pl_analysis: cell_seg = ut.load_region(data_dir + image_name + "_cell_segment")
	except IOError:
		print("Cannot load segments for {}".format(image_name))
		ow_segment = True
		ow_metric = True
		ow_figure = True

	try: 
		global_dataframe = pd.read_pickle('{}_global_metric.pkl'.format(data_dir + image_name))
		fibre_dataframe = pd.read_pickle('{}_fibre_metric.pkl'.format(data_dir + image_name))
		cell_dataframe = pd.read_pickle('{}_cell_metric.pkl'.format(data_dir + image_name))
			
	except IOError:
		print("Cannot load metrics for {}".format(image_name))
		ow_metric = True

	print(f"Overwrite options:\n ow_network = {ow_network}\n ow_segment = {ow_segment}\
		\n ow_metric = {ow_metric}\n ow_figure = {ow_figure}")

	start = time.time()


	if ow_network:

		ow_segment = True
	
		start_net = time.time()

		networks, networks_red, fibres = seg.network_extraction(image_shg, data_dir + image_name,
						sigma=sigma, p_denoise=p_denoise, threads=threads)

		ut.save_region(networks, data_dir + image_name + "_network")
		ut.save_region(networks_red, data_dir + image_name + "_network_reduced")
		ut.save_region(fibres, data_dir + image_name + "_fibre")

		end_net = time.time()

		print(f"TOTAL NETWORK EXTRACTION TIME = {round(end_net - start_net, 3)} s")

	if ow_segment:

		ow_metric = True
		ow_figure = True

		start_seg = time.time()

		ow_metric = True
		ow_figure = True

		networks = ut.load_region(data_dir + image_name + "_network")
		networks_red = ut.load_region(data_dir + image_name + "_network_reduced")

		fibre_net_seg = seg.fibre_segmentation(image_shg, networks, networks_red)

		if pl_analysis:

			cell_seg, fibre_col_seg = seg.cell_segmentation(image_shg, image_pl, image_tran)
			ut.save_region(cell_seg, '{}_cell_segment'.format(filename))

			fibre_seg = seg.hysteresis_segmentation(image_shg, fibre_col_seg, fibre_net_seg, 400, 0.075)
			ut.save_region(fibre_seg, '{}_fibre_segment'.format(filename))

		else:
			ut.save_region(fibre_net_seg, '{}_fibre_segment'.format(filename))

		end_seg = time.time()

		print(f"TOTAL SEGMENTATION TIME = {round(end_seg - start_seg, 3)} s")

	if ow_metric:
		
		start_met = time.time()

		"Load networks and segments"
		fibre_seg = ut.load_region(data_dir + image_name + "_fibre_segment")
		networks = ut.load_region(data_dir + image_name + "_network")
		networks_red = ut.load_region(data_dir + image_name + "_network_reduced")
		fibres = ut.load_region(data_dir + image_name + "_fibre")
		
		"Form nematic and structure tensors for each pixel"
		shg_n_tensor = form_nematic_tensor(image_shg, sigma=sigma)
		shg_j_tensor = form_structure_tensor(image_shg, sigma=sigma)

		"Perform anisotropy analysis on each pixel"
		shg_pix_n_anis, shg_pix_n_angle, shg_pix_n_energy = an.tensor_analysis(shg_n_tensor)
		shg_pix_j_anis, shg_pix_j_angle, shg_pix_j_energy = an.tensor_analysis(shg_j_tensor)
	
		print("Performing fibre segment analysis")

		"Analyse fibre network"
		net_res = seg.fibre_segment_analysis(image_shg, networks, networks_red, fibres, 
						fibre_seg, shg_j_tensor, shg_pix_j_anis, shg_pix_j_angle)

		(fibre_angle_sdi, fibre_anis, fibre_pix_anis,
		fibre_areas, fibre_linear, fibre_eccent, fibre_density, fibre_coverage,
		fibre_mean, fibre_std, fibre_entropy, fibre_glcm_contrast, 
		fibre_glcm_homo, fibre_glcm_dissim, fibre_glcm_corr, fibre_glcm_energy, 
		fibre_glcm_IDM, fibre_glcm_variance, fibre_glcm_cluster, fibre_glcm_entropy,
		fibre_hu, network_degree, network_eigen, network_connect, fibre_waviness, 
		fibre_lengths, fibre_cross_link) = net_res
		
		filenames = pd.Series(['{}_fibre_segment.pkl'.format(filename)] * len(fibre_seg), name='File')
		fibre_id = pd.Series(np.arange(len(fibre_seg)), name='ID')

		fibre_metrics = np.stack([fibre_angle_sdi, fibre_anis, 
			fibre_pix_anis, fibre_areas, fibre_linear, fibre_eccent, fibre_density,
			fibre_coverage, fibre_mean, fibre_std, fibre_entropy, 
			fibre_glcm_contrast, fibre_glcm_homo, fibre_glcm_dissim, fibre_glcm_corr,
			fibre_glcm_energy, fibre_glcm_IDM, fibre_glcm_variance, fibre_glcm_cluster, 
			fibre_glcm_entropy,network_degree, network_eigen, network_connect,
			fibre_waviness, fibre_lengths, fibre_cross_link], axis=-1)
		
		if pl_analysis:

			cell_seg = ut.load_region(data_dir + image_name + "_cell_segment")

			"Form nematic and structure tensors for each pixel"
			pl_n_tensor = form_nematic_tensor(image_pl, sigma=sigma)
			pl_j_tensor = form_structure_tensor(image_pl, sigma=sigma)

			"Perform anisotropy analysis on each pixel"
			pl_pix_n_anis, pl_pix_n_angle, pl_pix_n_energy = an.tensor_analysis(pl_n_tensor)
			pl_pix_j_anis, pl_pix_j_angle, pl_pix_j_energy = an.tensor_analysis(pl_j_tensor)

			print("Performing cell segment analysis")

			(_, _, _, _, _, _, _, muscle_glcm_contrast, muscle_glcm_homo, muscle_glcm_dissim, 
			muscle_glcm_corr, muscle_glcm_energy, muscle_glcm_IDM, muscle_glcm_variance, 
			muscle_glcm_cluster, muscle_glcm_entropy, _, _, _,
			_, _) = seg.cell_segment_analysis(image_pl, fibre_seg, pl_j_tensor, 
						pl_pix_j_anis, pl_pix_j_angle)

			muscle_metrics = np.stack([muscle_glcm_contrast, muscle_glcm_homo, 
			 	muscle_glcm_dissim, muscle_glcm_corr, muscle_glcm_energy, muscle_glcm_IDM, 
			 	muscle_glcm_variance, muscle_glcm_cluster, muscle_glcm_entropy], axis=-1)
			
			(cell_angle_sdi, cell_anis, cell_pix_anis, cell_areas, cell_mean, 
			cell_std, cell_entropy, cell_glcm_contrast, 
			cell_glcm_homo, cell_glcm_dissim, cell_glcm_corr, cell_glcm_energy, 
			cell_glcm_IDM, cell_glcm_variance, cell_glcm_cluster, cell_glcm_entropy,
			cell_linear, cell_eccent, cell_density, cell_coverage, 
			cell_hu) = seg.cell_segment_analysis(image_pl, cell_seg, pl_j_tensor, 
						pl_pix_j_anis, pl_pix_j_angle)

			filenames = pd.Series(['{}_cell_segment.pkl'.format(filename)] * len(cell_seg), name='File')		
			cell_id = pd.Series(np.arange(len(cell_seg)), name='ID')

			cell_metrics = np.stack([cell_angle_sdi, cell_anis, cell_pix_anis, cell_areas, 
						cell_mean, cell_std, cell_entropy, 
						cell_glcm_contrast, cell_glcm_homo, cell_glcm_dissim, cell_glcm_corr, 
						cell_glcm_energy, cell_glcm_IDM, cell_glcm_variance, cell_glcm_cluster,
						cell_glcm_entropy, cell_linear, cell_eccent, 
						cell_density, cell_coverage], axis=-1)
			cell_metrics = np.concatenate((cell_metrics, cell_hu), axis=-1)

		else:
			filenames = pd.Series(['{}_cell_segment.pkl'.format(filename)], name='File')		
			cell_id = pd.Series(np.arange(1), name='ID')
			cell_metrics = np.full_like(np.ones((1, len(cell_columns))), None)
			muscle_metrics = np.full_like(np.ones((1, 9)), None)

		fibre_metrics = np.concatenate((fibre_metrics, fibre_hu, muscle_metrics), axis=-1)

		fibre_dataframe = pd.DataFrame(data=fibre_metrics, columns=fibre_columns + muscle_columns)
		fibre_dataframe = pd.concat((filenames, fibre_id, fibre_dataframe), axis=1)
		fibre_dataframe.to_pickle('{}_fibre_metric.pkl'.format(filename))

		cell_dataframe = pd.DataFrame(data=cell_metrics, columns=cell_columns)
		cell_dataframe = pd.concat((filenames, cell_id, cell_dataframe), axis=1)
		cell_dataframe.to_pickle('{}_cell_metric.pkl'.format(filename))

		print("Performing Global Image analysis")

		fibre_binary = seg.create_binary_image(fibre_seg, image_shg.shape)
		global_binary = np.where(fibre_binary, 1, 0)
		global_segment = regionprops(global_binary)[0]

		global_columns = ['No. Fibres'] + fibre_columns[:-5]
		global_columns += muscle_columns
		global_columns += ['No. Cells'] + cell_columns[:-5]


		(__, global_angle_sdi, global_anis, global_pix_anis, 
		global_area, global_linear, global_eccent, global_density, 
		global_coverage, global_mean, global_std, global_entropy, global_glcm_contrast, 
		global_glcm_homo, global_glcm_dissim, global_glcm_corr, global_glcm_energy,
		global_glcm_IDM, global_glcm_variance, global_glcm_cluster, global_glcm_entropy,
		global_hu) = seg.segment_analysis(image_shg, global_segment, shg_j_tensor, 
						shg_pix_j_anis, shg_pix_j_angle)

		global_fibre_area = np.mean(fibre_areas)
		global_fibre_coverage = np.sum(fibre_binary) / image_shg.size
		global_fibre_linear = np.mean(fibre_linear)
		global_fibre_eccent = np.mean(fibre_eccent)
		global_fibre_density = np.mean(image_shg[np.where(fibre_binary)])
		global_fibre_hu_1 = np.mean(fibre_hu[:, 0])
		global_fibre_hu_2 = np.mean(fibre_hu[:, 1])

		global_fibre_waviness = np.nanmean(fibre_waviness)
		global_fibre_lengths = np.nanmean(fibre_lengths)
		global_fibre_cross_link = np.nanmean(fibre_cross_link)

		global_network_degree = np.nanmean(network_degree)
		global_network_eigen = np.nanmean(network_eigen)
		global_network_connect = np.nanmean(network_connect)
		global_nfibres = len(ut.flatten_list(fibres))

		global_fibre_metrics = np.stack([(global_nfibres),
					global_angle_sdi, global_anis, global_pix_anis,
					global_fibre_area, global_fibre_linear, global_fibre_eccent,
					global_fibre_density, global_fibre_coverage, global_mean, global_std, 
					global_entropy, global_glcm_contrast, 
					global_glcm_homo, global_glcm_dissim, global_glcm_corr, global_glcm_energy,
					global_glcm_IDM, global_glcm_variance, global_glcm_cluster, 
					global_glcm_entropy, global_network_degree,
					global_network_eigen, global_network_connect,
					global_fibre_waviness, global_fibre_lengths, global_fibre_cross_link,
					global_fibre_hu_1, global_fibre_hu_2], axis=0)

		if pl_analysis: 

			(__, _, _, _, _, _, _, _, _, _, _, _, global_glcm_contrast, 
			global_glcm_homo, global_glcm_dissim, global_glcm_corr, global_glcm_energy,
			global_glcm_IDM, global_glcm_variance, global_glcm_cluster, global_glcm_entropy,
			_) = seg.segment_analysis(image_pl, global_segment, shg_j_tensor, 
							shg_pix_j_anis, shg_pix_j_angle)

			global_muscle_metrics = np.stack([global_glcm_contrast, global_glcm_homo, 
				global_glcm_dissim, global_glcm_corr, global_glcm_energy,
				global_glcm_IDM, global_glcm_variance, global_glcm_cluster, global_glcm_entropy], axis=0)

			cell_binary = seg.create_binary_image(cell_seg, image_pl.shape)
			global_binary = np.where(cell_binary, 1, 0)
			global_segment = regionprops(global_binary)[0]

			(__, global_cell_angle_sdi, global_cell_anis, global_cell_pix_anis, global_cell_area,
			global_cell_linear, global_cell_eccent, global_cell_density, global_cell_coverage, 
			global_cell_mean, global_cell_std, global_cell_entropy, global_cell_glcm_contrast,
			global_cell_glcm_homo, global_cell_glcm_dissim, global_cell_glcm_corr, 
			global_cell_glcm_energy, global_cell_glcm_IDM, global_cell_glcm_variance, 
			global_cell_glcm_cluster, global_cell_glcm_entropy, 
			global_cell_hu) = seg.segment_analysis(image_pl, global_segment, pl_j_tensor, 
						pl_pix_j_anis, pl_pix_j_angle)

			global_ncells = len(cell_seg)

			global_cell_metrics = np.stack([(global_ncells),
						global_cell_angle_sdi, global_cell_anis, global_cell_pix_anis,
						global_cell_area, global_cell_mean, global_cell_std, 
						global_cell_entropy, global_cell_glcm_contrast,
						global_cell_glcm_homo, global_cell_glcm_dissim, global_cell_glcm_corr, 
						global_cell_glcm_energy, global_cell_glcm_IDM, global_cell_glcm_variance, 
						global_cell_glcm_cluster, global_cell_glcm_entropy,
						global_cell_linear, global_cell_eccent, global_cell_density,
						global_cell_coverage, global_cell_hu[0], global_cell_hu[1]], axis=0)

		else: 
			global_cell_metrics = np.full_like(np.ones((len(['No. Cells'] + cell_columns[:-5]))), None)
			global_muscle_metrics = np.full_like(np.ones((1, 9)), None)


		global_metrics = np.concatenate((global_fibre_metrics, global_muscle_metrics, 
			global_cell_metrics), axis=-1)
		filenames = pd.Series('{}_global_segment.pkl'.format(filename), name='File')
		global_metrics = np.expand_dims(global_metrics, 0)

		global_dataframe = pd.DataFrame(data=global_metrics, columns=global_columns)
		global_dataframe = pd.concat((filenames, global_dataframe), axis=1)
		global_dataframe.to_pickle('{}_global_metric.pkl'.format(filename))
		ut.save_region(global_segment, '{}_global_segment'.format(filename))

		end_met = time.time()

		print(f"TOTAL METRIC TIME = {round(end_met - start_met, 3)} s")

	if ow_figure:

		start_fig = time.time()
	
		networks =  ut.load_region(data_dir + image_name + "_network")
		fibre_seg = ut.load_region(data_dir + image_name + "_fibre_segment")
		fibres = ut.load_region(data_dir + image_name + "_fibre")
		fibres = ut.flatten_list(fibres)
		
		tensor_image = create_tensor_image(image_shg)
		network_image = create_network_image(image_shg, networks)
		fibre_image = create_network_image(image_shg, fibres, 1)
		fibre_region_image = create_region_image(image_shg, fibre_seg)

		create_figure(image_shg, fig_dir + image_name + '_SHG', cmap='binary_r')
		create_figure(tensor_image, fig_dir + image_name + '_tensor')
		create_figure(network_image, fig_dir + image_name + '_network')
		create_figure(fibre_image, fig_dir + image_name + '_fibre')
		create_figure(fibre_region_image, fig_dir + image_name + '_fibre_seg')

		if pl_analysis:
			cell_seg = ut.load_region(data_dir + image_name + "_cell_segment")
			cell_region_image = create_region_image(image_pl, cell_seg)
			create_figure(image_pl, fig_dir + image_name + '_PL', cmap='binary_r')
			create_figure(image_tran, fig_dir + image_name + '_trans', cmap='binary_r')
			create_figure(cell_region_image, fig_dir + image_name + '_cell_seg')

		end_fig = time.time()

		print(f"TOTAL FIGURE TIME = {round(end_fig - start_fig, 3)} s")
		
	end = time.time()

	print(f"TOTAL ANALYSIS TIME = {round(end - start, 3)} s")

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
	parser.add_argument('--ow_segment', action='store_true', help='Toggles overwrite image segmentation')
	parser.add_argument('--ow_network', action='store_true', help='Toggles overwrite network extraction')
	parser.add_argument('--ow_figure', action='store_true', help='Toggles overwrite figures')
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

	files, prefixes = ut.get_image_lists(input_files)

	print(input_files, files, prefixes)

	cell_database = pd.DataFrame()
	segment_database = pd.DataFrame()
	global_database = pd.DataFrame()
   
	for i, input_file_names in enumerate(files):

		image_path = '/'.join(prefixes[i].split('/')[:-1])

		data_global, data_segment, data_cell = analyse_image(input_file_names, 
			prefixes[i], image_path, sigma=args.sigma, 
			ow_metric=args.ow_metric, ow_segment=args.ow_segment, 
			ow_network=args.ow_network, ow_figure=args.ow_figure, 
			threads=args.threads)

		global_database = pd.concat([global_database, data_global])
		segment_database = pd.concat([segment_database, data_segment])
		cell_database = pd.concat([cell_database, data_cell])

		print(image_path)
		print("Global Image Analysis Metrics:")
		print(data_global.iloc[0])


	if args.save_db != None: 
		global_database.to_pickle('{}.pkl'.format(args.save_db))
		global_database.to_excel('{}.xls'.format(args.save_db))
		
		segment_database.to_pickle('{}_fibre.pkl'.format(args.save_db))
		segment_database.to_excel('{}_fibre.xls'.format(args.save_db))

		cell_database.to_pickle('{}_cell.pkl'.format(args.save_db))
		cell_database.to_excel('{}_cell.xls'.format(args.save_db))

		
