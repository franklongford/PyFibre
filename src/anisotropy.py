"""
ColECM: Collagen ExtraCellular Matrix Simulation
EXPERIMENTAL ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 10/08/2018

Last Modified: 10/08/2018
"""

import numpy as np
import scipy as sp
from scipy.ndimage import filters, imread

from skimage import io, img_as_float, exposure, data, color, restoration, feature, measure
from skimage.filters import threshold_otsu, try_all_threshold
from skimage.transform import swirl
from skimage.color import label2rgb

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys, os, time

import utilities as ut
from image_tools import (fourier_transform_analysis, print_fourier_results, select_samples,
						form_nematic_tensor, nematic_tensor_analysis, smart_nematic_tensor_analysis,
						print_anis_results, derivatives, int_clustering, spec_clustering)


def set_HSB(image, hue, saturation=1, brightness=1):
	""" Add color of the given hue to an RGB image.

	By default, set the saturation to 1 so that the colors pop!
	"""
	hsv = color.rgb2hsv(image)
	hsv[..., 0] = hue
	hsv[..., 1] = saturation
	#hsv[..., 2] = brightness

	return color.hsv2rgb(hsv)


def test_analysis(size=None, sigma=None):

	n = 100
	image_grid = np.mgrid[:n, :n]
	for i in range(2): image_grid[i] -= n * np.array(2 * image_grid[i] / n, dtype=int)
	image_grid = np.fft.fftshift(np.sqrt(np.sum(image_grid**2, axis=0)))
	#test_image = np.sin(n * np.pi / image_grid ) * np.cos(n * np.pi / image_grid)
	test_image = np.sin(10 * np.pi * image_grid / n ) * np.cos(10 * np.pi * image_grid / n)
	#test_image = np.zeros((n, n))
	#for i in range(10): test_image += np.eye(n, n, k=5-i)
	#for i in range(10): test_image += np.rot90(np.eye(n, n, k=5-i))

	#test_image = data.checkerboard()
	#test_image = swirl(test_image, rotation=0, strength=10, radius=120)

	fig = plt.figure()
	plt.imshow(test_image, cmap='Greys', interpolation='nearest', origin='lower')
	plt.axis("off")
	plt.colorbar()
	plt.savefig('test_image.png', bbox_inches='tight')
	plt.close()

	"""
	for i in range(2):
		plt.figure()
		plt.imshow(derivative[i], cmap='nipy_spectral', interpolation='nearest', origin='lower')
		plt.colorbar()
		plt.savefig('test_derivative_{}.png'.format(i), bbox_inches='tight')
		plt.close()
	"""
	n_tensor = form_nematic_tensor(test_image.reshape((1,) + test_image.shape), sigma=sigma)

	print(np.where(np.isnan(n_tensor)))
	
	"""
	for i in range(2):
		for j in range(2):
				plt.figure()
				plt.imshow(n_tensor[0,...,i,j], cmap='nipy_spectral', interpolation='nearest', origin='lower')
				plt.colorbar()
				plt.savefig('test_n_tensor_{}{}.png'.format(i, j), bbox_inches='tight')
				plt.close()

	plt.figure()
	plt.imshow(n_tensor[0,...,0,1] / (n_tensor[0,...,1,1] - n_tensor[0,...,0,0]), cmap='nipy_spectral', interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.savefig('test_n_tensor_diff.png'.format(i, j), bbox_inches='tight')
	plt.close()
	"""
	tot_q, tot_angle, tot_energy = nematic_tensor_analysis(n_tensor)

	picture = color.gray2rgb(test_image)
	hue = np.mod(tot_angle[0], 180) / 180
	saturation = tot_q[0] / tot_q[0].max()
	brightness = test_image / np.max(test_image)
	picture = set_HSB(picture, hue, saturation, brightness)

	plt.figure()
	plt.imshow(picture)
	plt.axis("off")
	plt.savefig('test_picture.png', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tot_q[0], cmap='binary_r', interpolation='nearest', origin='lower', vmin=0, vmax=1)
	plt.axis("off")
	plt.colorbar()
	plt.savefig('test_anisomap.png', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tot_energy[0], cmap='binary_r', interpolation='nearest', origin='lower')
	plt.axis("off")
	plt.colorbar()
	plt.savefig('test_energy.png', bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tot_angle[0], cmap='nipy_spectral', interpolation='nearest', origin='lower', vmin=-90, vmax=90)
	plt.axis("off")
	plt.colorbar()
	plt.savefig('test_anglemap.png', bbox_inches='tight')
	plt.close()

	
def analyse_image(current_dir, input_file_name, size=None, sigma=None):


	cmap = 'viridis'

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'

	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	image_name = input_file_name

	figure_name = ut.check_file_name(image_name, extension='tif')
	print(' {}'.format(figure_name))

	image_shg_orig = img_as_float(imread(image_name)).astype(np.float32)

	if image_shg_orig.ndim > 2: 
		image_shg_orig = np.sum(image_shg_orig / image_shg_orig.max(axis=-1), axis=0)

	image_shg = image_shg_orig / image_shg_orig.max()
	image_shg = filters.gaussian_filter(image_shg, sigma=sigma)

	#fig, ax = try_all_threshold(image_shg, figsize=(10, 8), verbose=False)
	#plt.savefig('{}{}_threshold.png'.format(fig_dir, figure_name))
	#plt.close()

	fig = plt.figure()
	plt.imshow(image_shg, cmap=cmap, interpolation='nearest', origin='lower')
	plt.savefig('{}{}.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	"""
	fig = plt.figure()
	plt.hist(image_shg.flatten(), bins=100)
	plt.savefig('{}{}_hist.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()
	"""

	image_shg = np.where(image_shg >= threshold_otsu(image_shg), image_shg, 0)

	n_tensor = form_nematic_tensor(image_shg.reshape((1,) + image_shg.shape), size=size)

	tot_q, tot_angle, tot_energy = nematic_tensor_analysis(n_tensor)

	print(' Mean pixel anistoropy = {:>6.4f}'.format(np.mean(tot_q)))

	q_filtered = np.where(tot_energy > 0.2, tot_q, 0)

	picture = color.gray2rgb(image_shg)
	hue = np.mod(tot_angle[0], 180) / 180
	saturation = tot_q[0] / tot_q[0].max()
	brightness = image_shg / np.max(image_shg)
	picture = set_HSB(picture, hue, saturation, brightness)

	plt.figure()
	plt.imshow(picture, vmin=0, vmax=1)
	plt.axis("off")
	plt.savefig('{}{}_picture.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tot_q[0], cmap='binary_r', interpolation='nearest', origin='lower', vmin=0, vmax=1)
	plt.colorbar()
	plt.savefig('{}{}_anisomap.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(np.where(tot_energy[0] > 0.2, tot_angle[0], -360), cmap='nipy_spectral', interpolation='nearest', origin='lower', vmin=-90, vmax=90)
	plt.colorbar()
	plt.savefig('{}{}_anglemap.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tot_energy[0], cmap='binary_r', interpolation='nearest', origin='lower')
	plt.axis("off")
	plt.colorbar()
	plt.savefig('{}{}_energymap.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Anisotropy Histogram')
	plt.hist(q_filtered[0].reshape(image_shg.shape[0]*image_shg.shape[1]), bins=100, density=True, label=figure_name, range=[0, 1])
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.ylim(0, 10)
	plt.legend()
	plt.savefig('{}{}_tot_aniso_hist.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Angular Histogram')
	plt.hist(tot_angle[0].reshape(image_shg.shape[0]*image_shg.shape[1]), 
				bins=100, density=True, label=figure_name, range=[-90, 90],
				weights=q_filtered[0].reshape(image_shg.shape[0]*image_shg.shape[1]))
	plt.xlabel(r'Angle')
	plt.xlim(-90, 90)
	plt.ylim(0, 0.05)
	plt.legend()
	plt.savefig('{}{}_tot_angle_hist.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	av_q, _ , _ = nematic_tensor_analysis(np.mean(n_tensor, axis=(1, 2)))

	print(' Mean image anistoropy = {:>6.4f}'.format(np.mean(av_q)))

	n_clusters = 3
	label_image, sorted_areas = int_clustering(tot_q[0], n_clusters, 0.3)
	cl_q = np.zeros((n_clusters))

	fig, ax = plt.subplots(figsize=(10, 6))
	image_label_overlay = label2rgb(label_image, image=tot_q[0])
	plt.imshow(image_label_overlay)
	for i, n in enumerate(sorted_areas):
		region =  measure.regionprops(label_image)[n]
		minr, minc, maxr, maxc = region.bbox
		rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
								fill=False, edgecolor='red', linewidth=2)
		ax.add_patch(rect)

		indices = np.mgrid[minr:maxr, minc:maxc]

		cl_q[i], _ , _ = nematic_tensor_analysis(np.mean(n_tensor[0][(indices[0], indices[1])], axis=(0, 1)))

	ax.set_axis_off()
	plt.savefig('{}{}_cluster_labels.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	print(' Mean cluster anistoropy = {:>6.4f}\n'.format(np.mean(cl_q)))

	"""
	mod_q = sp.misc.imresize(tot_q[0], 0.50) / 255.
	print(mod_q.shape, mod_q.size)
	labels = spec_clustering(mod_q, n_clusters)

	cl_q = np.zeros((n_clusters))
	plt.figure(figsize=(5, 5))
	plt.imshow(mod_q, cmap=plt.cm.gray, vmin=0, vmax=1)
	for l in range(n_clusters):
		plt.contour(labels == l, colors=[plt.cm.nipy_spectral(l / float(n_clusters))])
		cl_q[l] = np.mean(mod_q[labels == l])

	plt.axis("off")
	plt.title('Spectral clustering kmeans')
	plt.savefig('{}{}_cluster_kmeans.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	print(' Mean cluster anistoropy = {:>6.4f}\n'.format(np.mean(np.sort(cl_q)[-2:])))
	"""

	#tot_q = smart_nematic_tensor_analysis(n_tensor.reshape((1,) + n_tensor.shape))
	

	#angles, fourier_spec, sdi = fourier_transform_analysis(image_shg.reshape((1,) + image_shg.shape))
	#angles = angles[len(angles)//2:]
	#fourier_spec = 2 * fourier_spec[len(fourier_spec)//2:]
	#print_fourier_results(fig_dir, data_file_name, angles, fourier_spec, sdi)

	return np.mean(tot_q), np.mean(av_q), np.mean(cl_q)


def analyse_directory(current_dir, input_files, key=None):

	print()

	size = 1
	sigma = 1.0

	test_analysis(sigma=sigma)

	removed_files = []
	for file_name in input_files:
		if not (file_name.endswith('.tif')): removed_files.append(file_name)
		elif (file_name.find('display') != -1): removed_files.append(file_name)

		if key != None:
			if not (file_name.startswith(key)) and (file_name not in removed_files): 
				removed_files.append(file_name)

	for file_name in removed_files: input_files.remove(file_name)

	mean_tot_q = np.zeros(len(input_files))
	mean_av_q = np.zeros(len(input_files))
	mean_cl_q = np.zeros(len(input_files))

	for i, input_file_name in enumerate(input_files):
		mean_tot_q[i], mean_av_q[i], mean_cl_q[i] = analyse_image(current_dir, input_file_name, size=size, sigma=sigma)

	fig_dir = current_dir + '/fig/'
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)

	index = np.arange(len(input_files))
	bar_width = 0.25

	plt.figure()
	plt.title('Average Anisotropy')
	plt.bar(index, mean_tot_q, bar_width, align='center', label='pixel average')
	plt.bar(index + bar_width, mean_cl_q, bar_width, align='center', label='cluster average')
	plt.bar(index + 2 * bar_width, mean_av_q, bar_width, align='center', label='image average')
	plt.ylabel('Anisotropy')
	plt.title('Scores by person')
	plt.xticks(index + 1.5 * bar_width, input_files)
	plt.setp(plt.gca().get_xticklabels(), rotation=25, horizontalalignment='right')
	plt.legend()
	if key == None: plt.savefig('{}average_anis.png'.format(fig_dir), bbox_inches='tight')
	else: plt.savefig('{}average_anis_{}.png'.format(fig_dir, key), bbox_inches='tight')
	plt.close()
