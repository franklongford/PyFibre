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
from skimage.transform import swirl
from skimage.color import label2rgb, gray2rgb
from skimage.filters import threshold_otsu, hessian
from skimage.restoration import denoise_tv_chambolle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import utilities as ut
import image_tools as it
from extraction import FIRE



def plot_figures(fig_dir, fig_name, image, anis, angle, energy, cmap='viridis'):
	"""
	plot_figures(fig_name, fig_dir, image, anis, angle, energy, cmap='viridis')

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
	plt.imshow(anis, cmap='binary_r', interpolation='nearest', vmin=0, vmax=1)
	plt.colorbar()
	plt.savefig('{}{}_anisomap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(angle, cmap='nipy_spectral', interpolation='nearest', vmin=-90, vmax=90)
	plt.colorbar()
	plt.savefig('{}{}_anglemap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(energy, cmap='binary_r', interpolation='nearest')
	plt.axis("off")
	plt.colorbar()
	plt.savefig('{}{}_energymap.png'.format(fig_dir, fig_name), bbox_inches='tight')
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
	ax.imshow(image_label_overlay)
	#for r in rect: ax.add_patch(r)
	ax.set_axis_off()
	plt.savefig('{}{}_{}_labels.png'.format(fig_dir, fig_name, mode), bbox_inches='tight')
	plt.close()


def plot_graph(fig_dir, fig_name, image, graph, mode):

	# draw image
	plt.imshow(image, cmap='gray')

	# draw edges by pts
	for (s,e) in graph.edges():
		ps = graph[s][e]['pts']
		plt.plot(ps[:,1], ps[:,0], 'green')

	# draw node by o
	node, nodes = graph.node, graph.nodes()
	ps = np.array([node[i]['o'] for i in nodes])
	plt.plot(ps[:,1], ps[:,0], 'r.')

	# title and show
	plt.title('Build Graph')
	plt.savefig('{}{}_{}_graph.png'.format(fig_dir, fig_name, mode), bbox_inches='tight')
	plt.close()


def test_analysis(current_dir, size=None, sigma=None, ow_anis=False):
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

	N = 200

	fig_dir = current_dir + '/'
	fig_names = [ 'test_image_line', 
				'test_image_cross', 'test_image_noise', 
				'test_image_checker', 'test_image_rings',
				'test_image_fibres_flex', 'test_image_fibres_stiff']

	ske_clus = np.zeros(len(fig_names))
	ske_path = np.zeros(len(fig_names))
	ske_solid = np.zeros(len(fig_names))
	ske_lin = np.zeros(len(fig_names))
	ske_curve = np.zeros(len(fig_names))
	ske_cover = np.zeros(len(fig_names))
	mean_img_anis = np.zeros(len(fig_names))
	mean_pix_anis = np.zeros(len(fig_names))
	mean_ske_anis = np.zeros(len(fig_names))

	for n, fig_name in enumerate(fig_names):

		if fig_name == 'test_image_rings':
			image_grid = np.mgrid[:N, :N]
			for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
			image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
			test_image = np.sin(10 * np.pi * image_grid / N ) * np.cos(10 * np.pi * image_grid / N)
		elif fig_name == 'test_image_circle':
			image_grid = np.mgrid[:N, :N]
			for i in range(2): image_grid[i] -= N * np.array(2 * image_grid[i] / N, dtype=int)
			image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
			test_image = 1 - ut.gaussian(image_grid, N / 4, 5)
		elif fig_name == 'test_image_line':
			test_image = np.zeros((N, N))
			for i in range(10): test_image += np.eye(N, N, k=5-i)
		elif fig_name == 'test_image_cross':
			test_image = np.zeros((N, N))
			for i in range(10): test_image += np.eye(N, N, k=5-i)
			for i in range(10): test_image += np.rot90(np.eye(N, N, k=5-i))
			test_image = np.where(test_image != 0, 1, 0)
			FIRE(test_image)
		elif fig_name == 'test_image_noise':
			test_image = np.random.random((N, N))
		elif fig_name == 'test_image_checker':
			test_image = data.checkerboard()
			test_image = swirl(test_image, rotation=0, strength=10, radius=120)
		elif fig_name == 'test_image_fibres_stiff':
			test_image = np.load('col_2D_stiff_data.npy')[0]
		elif fig_name == 'test_image_fibres_flex':
			test_image = np.load('col_2D_flex_data.npy')[0]

		res = analyse_image(current_dir, fig_name, test_image, sigma=sigma, ow_anis=ow_anis, mode='test')
		(ske_clus[n], ske_lin[n], ske_cover[n], ske_curve[n], ske_solid[n],
			mean_ske_anis[n], mean_pix_anis[n], mean_img_anis[n]) = res

		print(' Skeleton Clustering = {:>6.4f}'.format(ske_clus[n]))
		print(' Skeleton Linearity = {:>6.4f}'.format(ske_lin[n]))
		print(' Skeleton Coverage = {:>6.4f}'.format(ske_cover[n]))
		print(' Skeleton Solidity = {:>6.4f}'.format(ske_solid[n]))
		print(' Skeleton Curvature = {:>6.4f}'.format(ske_curve[n]))
		print(' Skeleton Anistoropy = {:>6.4f}'.format(mean_ske_anis[n]))
		print(' Total Pixel anistoropy = {:>6.4f}'.format(mean_pix_anis[n]))
		print(' Total Image anistoropy = {:>6.4f}\n'.format(mean_img_anis[n]))

	x_labels = fig_names
	col_len = len(max(x_labels, key=len))

	predictor = predictor_metric(ske_clus, ske_lin, ske_cover, ske_solid,
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

	
def analyse_image(current_dir, input_file_name, image, size=None, sigma=None, n_clusters=10, ow_anis=False, mode='SHG'):

	cmap = 'viridis'

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'
	image_name = input_file_name

	fig_name = ut.check_file_name(image_name, extension='tif')

	print(' {}'.format(fig_name))

	if not ow_anis and os.path.exists(data_dir + fig_name + '.npy'):
		averages = ut.load_npy(data_dir + fig_name)
	else:
		image = it.prepare_image_shg(image, sigma=sigma, tf=True)

		fig, ax = plt.subplots(figsize=(10, 6))
		plt.imshow(image, cmap=cmap, interpolation='nearest')
		ax.set_axis_off()
		plt.savefig('{}{}_orig.png'.format(fig_dir, fig_name), bbox_inches='tight')
		plt.close()

		if mode == 'SHG':
			image_shg = it.prepare_image_shg(image, clip_limit=0.05, tf=True)

			fig, ax = plt.subplots(figsize=(10, 6))
			plt.imshow(image_shg, cmap=cmap, interpolation='nearest')
			ax.set_axis_off()
			plt.savefig('{}{}_recon.png'.format(fig_dir, fig_name), bbox_inches='tight')
			plt.close()
	
		else: image_shg = image

		n_tensor = it.form_nematic_tensor(image_shg, sigma=sigma)
		j_tensor = it.form_structure_tensor(image_shg, sigma=sigma)
		H_tensor = it.form_hessian_tensor(image_shg, sigma=sigma)

		"Perform anisotropy analysis on each pixel"

		pix_n_anis, pix_n_angle, pix_n_energy = it.tensor_analysis(n_tensor)
		pix_j_anis, pix_j_angle, pix_j_energy = it.tensor_analysis(j_tensor)
		pix_H_anis, pix_H_angle, pix_H_energy = it.tensor_analysis(H_tensor)

		plot_figures(fig_dir, fig_name, image_shg, pix_j_anis, pix_j_angle, pix_j_energy, cmap='viridis')

		fig, ax = plt.subplots(figsize=(10, 6))
		plt.imshow(np.where(pix_n_anis > threshold_otsu(pix_n_anis), pix_n_anis, 0), cmap='Greys', interpolation='nearest')
		plt.colorbar()
		ax.set_axis_off()
		plt.savefig('{}{}_pix_nematic.png'.format(fig_dir, fig_name), bbox_inches='tight')
		plt.close()
		
		gauss_curvature, mean_curvature = it.get_curvature(j_tensor, H_tensor)
		#mean_curvature = np.trace(H_tensor, axis1=-2, axis2=-1)

		mean_curvature = abs(mean_curvature)
		fig, ax = plt.subplots(figsize=(10, 6))
		plt.imshow(np.where(mean_curvature > threshold_otsu(mean_curvature), mean_curvature, 0), cmap='Greys', interpolation='nearest')
		plt.colorbar()
		ax.set_axis_off()
		plt.savefig('{}{}_mean_curvature.png'.format(fig_dir, fig_name), bbox_inches='tight')
		plt.close()

		"""
		pix_H_energy = abs(pix_H_energy)
		fig, ax = plt.subplots(figsize=(10, 6))
		plt.imshow(np.where(pix_H_energy > threshold_otsu(pix_H_energy), 1, 0), cmap='Greys', interpolation='nearest', origin='lower')
		plt.colorbar()
		ax.set_axis_off()
		plt.savefig('{}{}_H_energy.png'.format(fig_dir, fig_name), bbox_inches='tight')
		plt.close()
		"""

		"Perform anisotropy analysis on whole image"

		img_anis, _ , _ = it.tensor_analysis(np.mean(n_tensor, axis=(0, 1)))
		img_H_anis, _ , _ = it.tensor_analysis(np.mean(H_tensor, axis=(0, 1)))

		"Extract main collagen network using mean curvature"

		(label_image, sorted_areas, net_path, 
			clustering, main_network, graph) = it.network_extraction(image_shg, n_clusters)
		(net_area, net_anis, net_linear, net_curve,
			region_anis, coverage, solidity) = it.network_analysis(label_image, sorted_areas, j_tensor, 
																	pix_j_anis, mean_curvature)

		plt.imshow(np.where(main_network != 0, 1, 0), cmap='Greys')
		plt.savefig('{}{}_{}_network.png'.format(fig_dir, fig_name, 'skeleton'), bbox_inches='tight')
		plt.close()

		if mode == 'test':
			plot_graph(fig_dir, fig_name, image_shg, graph, 'skeleton')

		plot_labeled_figure(fig_dir, fig_name, image_shg, label_image, sorted_areas, mode)

		anisotropy = np.average(net_anis, weights=net_area)
		linearity = np.average(net_linear, weights=net_area)
		pix_anis = np.average(region_anis, weights=net_area)
		solidity = np.average(solidity, weights=net_area)
		curvature = np.average(net_curve, weights=net_area)

		averages = (clustering, linearity, coverage, curvature,
					solidity, anisotropy, pix_anis, img_anis)

		ut.save_npy(data_dir + fig_name, averages)

	return averages

def predictor_metric(clus, lin, cover, solid, region_anis, pix_anis, img_anis):

	predictor = np.sqrt(clus**2 + lin**2 + region_anis**2 + pix_anis**2) / np.sqrt(4)
	#predictor = np.sqrt(ske_clus**2 + ske_anis**2 + con_anis**2 + pix_anis**2 + img_anis**2) / np.sqrt(5)
	#predictor = np.sqrt(ske_clus**2 + ske_anis**2 + con_anis**2 + pix_anis**2) / np.sqrt(4)
	#predictor =  (ske_clus + ske_anis + pix_anis) / 3
	#predictor =  (ske_clus + con_clus) / 2 * np.sqrt(ske_anis**2 + con_anis**2 + pix_anis**2) / np.sqrt(3)

	return predictor


def analyse_directory(current_dir, input_files, key=None, ow_anis=False):

	print()

	size = 2
	sigma = 0.5

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'

	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	if ('-test' in sys.argv): test_analysis(current_dir, sigma=sigma, ow_anis=ow_anis)

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
		ske_curve = np.zeros(len(input_files))
		ske_cover = np.zeros(len(input_files))
		mean_img_anis = np.zeros(len(input_files))
		mean_pix_anis = np.zeros(len(input_files))
		mean_ske_anis = np.zeros(len(input_files))

		for i, input_file_name in enumerate(input_files):
			image = it.load_tif(input_file_name)
			res = analyse_image(current_dir, input_file_name, image, size=size, 
								sigma=sigma, ow_anis=ow_anis, mode=mode)
			(ske_clus[i], ske_lin[i], ske_cover[i], ske_curve[i], ske_solid[i],
				mean_ske_anis[i], mean_pix_anis[i], mean_img_anis[i]) = res

			print(' Skeleton Clustering = {:>6.4f}'.format(ske_clus[i]))
			print(' Skeleton Linearity = {:>6.4f}'.format(ske_lin[i]))
			print(' Skeleton Coverage = {:>6.4f}'.format(ske_cover[i]))
			print(' Skeleton Solidity = {:>6.4f}'.format(ske_solid[i]))
			print(' Skeleton Curvature = {:>6.4f}'.format(ske_curve[i]))
			print(' Skeleton Anistoropy = {:>6.4f}'.format(mean_ske_anis[i]))
			print(' Total Pixel anistoropy = {:>6.4f}'.format(mean_pix_anis[i]))
			print(' Total Image anistoropy = {:>6.4f}\n'.format(mean_img_anis[i]))

		x_labels = [ut.check_file_name(image_name, extension='tif') for image_name in input_files]
		col_len = len(max(x_labels, key=len))

		predictor = predictor_metric(ske_clus, ske_lin, ske_cover, ske_solid,
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
