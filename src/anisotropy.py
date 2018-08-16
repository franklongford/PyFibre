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
from skimage import io, img_as_float, exposure

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys, os

import utilities as ut
from analysis import (fourier_transform_analysis, print_fourier_results, select_samples,
						form_nematic_tensor, nematic_tensor_analysis, print_anis_results,
						derivatives)


def analyse_image(current_dir, input_file_name):

	cmap = 'viridis'
	size = 10
	sigma = 2

	fig_dir = current_dir + '/fig/'
	data_dir = current_dir + '/data/'

	if not os.path.exists(fig_dir): os.mkdir(fig_dir)
	if not os.path.exists(data_dir): os.mkdir(data_dir)

	image_name = input_file_name
	image_shg_orig = img_as_float(imread(image_name)).astype(np.float32)

	if image_shg_orig.ndim > 2: 
		image_shg_orig = np.sum(image_shg_orig, axis=0)

	image_shg = image_shg_orig
	#image_shg = filters.gaussian(image_shg_orig, sigma=sigma)
	image_shg = exposure.equalize_adapthist(image_shg, clip_limit=0.03)

	figure_name = ut.check_file_name(image_name, extension='tif')

	print(' {}'.format(figure_name))

	fig = plt.figure()
	plt.imshow(image_shg, cmap=cmap, interpolation='nearest', origin='lower')
	plt.savefig('{}{}.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	if os.path.exists(data_dir + figure_name + '_tot_analysis.npy'):
		print(' Loading nematic tensor data')
		tot_q, tot_angle = ut.load_npy(data_dir + figure_name + '_tot_analysis')
		av_q, av_angle = ut.load_npy(data_dir + figure_name + '_av_analysis')
	else:
		print(' Creating nematic tensor data')
		derivative = derivatives(image_shg)
		n_tensor = form_nematic_tensor(derivative[0].reshape((1,) + image.shape),
									   derivative[1].reshape((1,) + image.shape),
									   size=size)

		tot_q, tot_angle, av_q, av_angle = nematic_tensor_analysis(n_tensor)

		ut.save_npy(data_dir + figure_name + '_tot_analysis', np.array((tot_q, tot_angle)))
		ut.save_npy(data_dir + figure_name + '_av_analysis', np.array((av_q, av_angle)))

	lim = 1.5 * np.std(image_shg)
	q_filtered = np.where(image_shg / image_shg.max() >= lim, tot_q, -1)
	angle_filtered = np.where(image_shg / image_shg.max() >= lim, tot_angle, -360)

	plt.figure()
	plt.imshow(q_filtered[0], cmap='binary_r', interpolation='nearest', origin='lower', vmin=0, vmax=1)
	plt.colorbar()
	plt.savefig('{}{}_anisomap.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(angle_filtered[0], cmap='nipy_spectral', interpolation='nearest', origin='lower', vmin=-45, vmax=45)
	plt.colorbar()
	plt.savefig('{}{}_anglemap.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Anisotropy Histogram')
	plt.hist(q_filtered[0].reshape(image.shape[0]*image.shape[1]), bins=100, density=True, label=figure_name, range=[0, 1])
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.ylim(0, 10)
	plt.legend()
	plt.savefig('{}{}_tot_aniso_hist.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Angular Histogram')
	plt.hist(angle_filtered[0].reshape(image.shape[0]*image.shape[1]), bins=100, density=True, label=figure_name, range=[-45, 45])
	plt.xlabel(r'Angle')
	plt.xlim(-45, 45)
	plt.ylim(0, 0.05)
	plt.legend()
	plt.savefig('{}{}_tot_angle_hist.png'.format(fig_dir, figure_name), bbox_inches='tight')
	plt.close()

	print(' Mean pixel anistoropy = {:>6.4f}'.format(np.mean(tot_q)))
	print(' Mean image anistoropy = {:>6.4f}\n'.format(np.mean(av_q)))
	

	#angles, fourier_spec, sdi = fourier_transform_analysis(image_shg.reshape((1,) + image_shg.shape))
	#angles = angles[len(angles)//2:]
	#fourier_spec = 2 * fourier_spec[len(fourier_spec)//2:]
	#print_fourier_results(fig_dir, data_file_name, angles, fourier_spec, sdi)

	return np.mean(tot_q), np.mean(av_q) 


def analyse_directory(current_dir, input_files, key=None):

	print()

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

	for i, input_file_name in enumerate(input_files):
		mean_tot_q[i], mean_av_q[i] = analyse_image(current_dir, input_file_name)

	fig_dir = current_dir + '/fig/'
	if not os.path.exists(fig_dir): os.mkdir(fig_dir)

	index = np.arange(len(input_files))
	bar_width = 0.4

	plt.figure()
	plt.title('Average Anisotropy')
	plt.bar(index, mean_tot_q, bar_width, align='center', label='pixel average')
	plt.bar(index + bar_width, mean_av_q, bar_width, align='center', label='image average')
	plt.ylabel('Anisotropy')
	plt.title('Scores by person')
	plt.xticks(index + bar_width, input_files)
	plt.setp(plt.gca().get_xticklabels(), rotation=25, horizontalalignment='right')
	plt.legend()
	if key == None: plt.savefig('{}average_anis.png'.format(fig_dir), bbox_inches='tight')
	else: plt.savefig('{}average_anis_{}.png'.format(fig_dir, key), bbox_inches='tight')
	plt.close()
