"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import numpy as np
import scipy as sp
from scipy import signal, interpolate, ndimage
from scipy.ndimage import filters

from skimage import measure, transform, feature
from skimage.morphology import disk, square, label, convex_hull_image, closing
from skimage.filters import rank, threshold_otsu
from skimage.color import label2rgb

from sklearn.feature_extraction import image as image_fe
from sklearn.cluster import spectral_clustering

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3d
import matplotlib.animation as animation

import sys, os

import utilities as ut


def print_anis_results(fig_dir, fig_name, tot_q, tot_angle, av_q, av_angle):

	nframe = tot_q.shape[0]
	nxy = tot_q.shape[1]
	print('\n Mean image anistoropy = {:>6.4f}'.format(np.mean(av_q)))
	print('Mean pixel anistoropy = {:>6.4f}\n'.format(np.mean(tot_q)))

	plt.figure()
	plt.hist(av_q, bins='auto', density=True, label=fig_name, range=[0, 1])
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.legend()
	plt.savefig('{}{}_av_aniso_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.hist(av_angle, bins='auto', density=True, label=fig_name, range=[-45, 45])
	plt.xlabel(r'Anisotropy')
	plt.xlim(-45, 45)
	plt.legend()
	plt.savefig('{}{}_av_angle_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	"""
	plt.figure()
	plt.imshow(tot_q[0], cmap='binary_r', interpolation='nearest', origin='lower', vmin=0, vmax=1)
	plt.colorbar()
	plt.savefig('{}{}_anisomap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.imshow(tot_angle[0], cmap='nipy_spectral', interpolation='nearest', origin='lower', vmin=-45, vmax=45)
	plt.colorbar()
	plt.savefig('{}{}_anglemap.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()
	"""
	q_hist = np.zeros(100)
	angle_hist = np.zeros(100)

	for frame in range(nframe):
		q_hist += np.histogram(tot_q[frame].flatten(), bins=100, density=True, range=[0, 1])[0] / nframe
		angle_hist += np.histogram(tot_angle[frame].flatten(), bins=100, density=True, range=[-45, 45])[0] / nframe

	plt.figure()
	plt.title('Anisotropy Histogram')
	plt.plot(np.linspace(0, 1, 100), q_hist, label=fig_name)
	plt.xlabel(r'Anisotropy')
	plt.xlim(0, 1)
	plt.legend()
	plt.savefig('{}{}_tot_aniso_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()

	plt.figure()
	plt.title('Angular Histogram')
	plt.plot(np.linspace(-45, 45, 100), angle_hist, label=fig_name)
	plt.xlabel(r'Angle')
	plt.xlim(-45, 45)
	plt.legend()
	plt.savefig('{}{}_tot_angle_hist.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close()


def print_fourier_results(fig_dir, fig_name, angles, fourier_spec, sdi):

	print('\n Modal Fourier Amplitude  = {:>6.4f}'.format(angles[np.argmax(fourier_spec)]))
	print(' Fourier Amplitudes Range   = {:>6.4f}'.format(np.max(fourier_spec)-np.min(fourier_spec)))
	print(' Fourier Amplitudes Std Dev = {:>6.4f}'.format(np.std(fourier_spec)))
	print(' Fourier SDI = {:>6.4f}'.format(sdi))

	print(' Creating Fouier Angle Spectrum figure {}{}_fourier.png'.format(fig_dir, fig_name))
	plt.figure(11)
	plt.title('Fourier Angle Spectrum')
	plt.plot(angles, fourier_spec, label=fig_name)
	plt.xlabel(r'Angle (deg)')
	plt.ylabel(r'Amplitude')
	plt.xlim(-180, 180)
	plt.ylim(0, 1)
	plt.legend()
	plt.savefig('{}{}_fourier.png'.format(fig_dir, fig_name), bbox_inches='tight')
	plt.close('all')


def select_samples(full_set, area, n_sample):
	"""
	select_samples(full_set, area, n_sample)

	Selects n_sample random sections of image stack full_set

	Parameters
	----------

	full_set:  array_like (float); shape(n_frame, n_y, n_x)
		Full set of n_frame images

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	data_set:  array_like (float); shape=(n_sample, 2, n_y, n_x)
		Sampled areas

	indices:  array_like (float); shape=(n_sample, 2)
		Starting points for random selection of full_set

	"""
	
	if full_set.ndim == 2: full_set = full_set.reshape((1,) + full_set.shape)

	n_frame = full_set.shape[0]
	n_y = full_set.shape[1]
	n_x = full_set.shape[2]

	data_set = np.zeros((n_sample, n_frame, area, area))

	pad = area // 2

	indices = np.zeros((n_sample, 2), dtype=int)

	for n in range(n_sample):

		try: start_x = np.random.randint(pad, n_x - pad)
		except: start_x = pad
		try: start_y = np.random.randint(pad, n_y - pad) 
		except: start_y = pad

		indices[n][0] = start_x
		indices[n][1] = start_y

		data_set[n] = full_set[:, start_y-pad: start_y+pad, 
					  start_x-pad: start_x+pad]

	return data_set.reshape(n_sample * n_frame, area, area), indices


def derivatives(image, mode='cd'):

	derivative = np.zeros(((2,) + image.shape))
	if mode == 'cd': 
		derivative[0] += np.nan_to_num(np.gradient(image, edge_order=1, axis=-2))
		derivative[1] += np.nan_to_num(np.gradient(image, edge_order=1, axis=-1))

	elif mode == 'sobel':
		derivative[0] += np.nan_to_num(ndimage.sobel(image, axis=-2))
		derivative[1] += np.nan_to_num(ndimage.sobel(image, axis=-1))

	elif mode == 'bspl':
		x = np.arange(image.shape[0])
		y = np.arange(image.shape[1])

		spline = interpolate.RectBivariateSpline(x, y, image)
		image_bispl = spline.ev(*np.meshgrid(x, y))
		derivative = np.gradient(image_bispl)

		fig = plt.figure()
		plt.imshow(image_bispl, cmap='Greys', interpolation='nearest', origin='lower')
		plt.axis("off")
		plt.colorbar()
		plt.savefig('test_image_bispl.png', bbox_inches='tight')
		plt.close()

	return derivative

def form_nematic_tensor(image, sigma=None, size=None):
	"""
	form_nematic_tensor(dx_shg, dy_shg)

	Create local nematic tensor n for each pixel in dx_shg, dy_shg

	Parameters
	----------

	dx_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to x axis for each pixel

	dy_grid:  array_like (float); shape=(nframe, n_y, n_x)
		Matrix of derivative of image intensity with respect to y axis for each pixel

	Returns
	-------

	n_vector:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)	

	"""

	nframe = image.shape[0]
	dx_shg, dy_shg = derivatives(image, mode='cd')
	r_xy_2 = (dx_shg**2 + dy_shg**2)
	indicies = np.where(r_xy_2 > 0)

	nxx = np.zeros(dx_shg.shape)
	nyy = np.zeros(dx_shg.shape)
	nxy = np.zeros(dx_shg.shape)

	nxx[indicies] += dy_shg[indicies]**2 / r_xy_2[indicies]
	nyy[indicies] += dx_shg[indicies]**2 / r_xy_2[indicies]
	nxy[indicies] -= dx_shg[indicies] * dy_shg[indicies] / r_xy_2[indicies]


	if sigma != None:
		for frame in range(nframe):
			nxx[frame] = filters.gaussian_filter(nxx[frame], sigma=sigma)
			nyy[frame] = filters.gaussian_filter(nyy[frame], sigma=sigma)
			nxy[frame] = filters.gaussian_filter(nxy[frame], sigma=sigma)
	elif size != None:
		for frame in range(nframe):

			nxx[frame] = filters.maximum_filter(nxx[frame], size=size)
			nyy[frame] = filters.maximum_filter(nyy[frame], size=size)
			nxy[frame] = filters.maximum_filter(nxy[frame], size=size)

			#nxx[frame] = filters.uniform_filter(nxx[frame], size=size)
			#nyy[frame] = filters.uniform_filter(nyy[frame], size=size)
			#nxy[frame] = filters.uniform_filter(nxy[frame], size=size)

	#nxx, nxy, nyy = feature.structure_tensor(image[0])

	n_vector = np.stack((nxx, nxy, nxy, nyy), -1).reshape(nxx.shape + (2,2))

	return n_vector


def nematic_tensor_analysis(nem_vector):
	"""
	nematic_tensor_analysis(nem_vector)

	Calculates eigenvalues and eigenvectors of average nematic tensor over area^2 pixels for n_samples

	Parameters
	----------

	nem_vector:  array_like (float); shape(n_frame, n_y, n_x, 4)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	av_eigval:  array_like (float); shape=(n_frame, n_sample, 2)
		Eigenvalues of average nematic tensors for n_sample areas

	av_eigvec:  array_like (float); shape=(n_frame, n_sample, 2, 2)
		Eigenvectors of average nematic tensors for n_sample areas

	"""

	eig_val, eig_vec = np.linalg.eig(nem_vector)
	tot_q = eig_val.max(axis=-1) - eig_val.min(axis=-1)
	#tot_angle = np.arctan2(eig_vec[..., 0, 0], eig_vec[..., 0, 1]) / np.pi * 180
	tot_angle = np.arctan2(nem_vector[..., 1, 0], nem_vector[..., 1, 1]) / np.pi * 180
	tot_energy = np.trace(nem_vector, axis1=-2, axis2=-1)

	return tot_q, tot_angle, tot_energy


def spec_clustering(image, n_clusters=3):

	graph = image_fe.img_to_graph(image)
	labels = spectral_clustering(graph, n_clusters=n_clusters, assign_labels='kmeans', random_state=1)
	return labels.reshape(image.shape)


def clear_border(image):

	image[:, 0] = 0
	image[0, :] = 0
	image[:, -1] = 0
	image[-1, :] = 0

	return image

def int_clustering(image, n_clusters=3, red=0.5):

	#image = np.where(image >= threshold_otsu(image), 1, 0)
	#c_hull = convex_hull_image(image)

	image_trans = transform.rescale(image, red) / 255.
	thresh = threshold_otsu(image_trans)
	bw = closing(image_trans > thresh, square(3))
	# remove artifacts connected to image border
	cleared = clear_border(bw)

	# label image regions
	label_image = measure.label(cleared)
	label_image = transform.resize(label_image, image.shape, preserve_range=True, anti_aliasing=False)
	label_image = np.array(label_image, dtype=int)

	print(label_image.shape)
	
	areas = []
	for region in measure.regionprops(label_image): areas.append(region.area)
	sort_areas = np.argsort(areas)[-n_clusters:]

	return label_image, sort_areas


def smart_nematic_tensor_analysis(nem_vector, precision=1E-1):
	"""
	nematic_tensor_analysis(nem_vector)

	Calculates eigenvalues and eigenvectors of average nematic tensor over area^2 pixels for n_samples

	Parameters
	----------

	nem_vector:  array_like (float); shape(n_frame, n_y, n_x, 4)
		Flattened 2x2 nematic vector for each pixel in dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	av_eigval:  array_like (float); shape=(n_frame, n_sample, 2)
		Eigenvalues of average nematic tensors for n_sample areas

	av_eigvec:  array_like (float); shape=(n_frame, n_sample, 2, 2)
		Eigenvectors of average nematic tensors for n_sample areas

	"""

	tot_q, tot_angle, tot_energy = nematic_tensor_analysis(nem_vector)

	n_sample = nem_vector.shape[0]
	
	for n in range(n_sample):
		section = np.argmax(tot_q[n])

	def rec_search(nem_vector, q):

		image_shape = q.shape
		if image_shape[0] <= 2: return q

		for i in range(2):
			for j in range(2):
				vec_section = nem_vector[:,
									i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
									j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2 ]
				q_section = q[i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
							j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2]

				new_q, _ = nematic_tensor_analysis(np.mean(vector_map, axis=(1, 2)))
				old_q = np.mean(q_section)

				if abs(new_q - old_q) >= precision: q_section = rec_search(vec_section, q_section)
				else: q_section = np.ones(vec_section.shape[1:]) * new_q

				q[i * image_shape[0] // 2 : (i+1) * image_shape[0] // 2,
				  j * image_shape[1] // 2 : (j+1) * image_shape[1] // 2] = q_section

		return q

	for n in range(n_sample):
		vector_map = nem_vector[n]
		q0 = np.zeros(map_shape)

		print(vector_map.shape, np.mean(vector_map, axis=(1, 2)))

		av_q, _ = nematic_tensor_analysis(np.mean(vector_map, axis=(1, 2)))

		q0 += av_q
		q1 = rec_search(vector_map, q0)

		tot_q[n] = np.mean(np.unique(q1))

	return tot_q


def fourier_transform_analysis(image_shg):
	"""
	fourier_transform_analysis(image_shg, area, n_sample)

	Calculates fourier amplitude spectrum of over area^2 pixels for n_samples

	Parameters
	----------

	image_shg:  array_like (float); shape=(n_images, n_x, n_y)
		Array of images corresponding to each trajectory configuration

	area:  int
		Unit length of sample area

	n_sample:  int
		Number of randomly selected areas to sample

	Returns
	-------

	angles:  array_like (float); shape=(n_bins)
		Angles corresponding to fourier amplitudes

	fourier_spec:  array_like (float); shape=(n_bins)
		Average Fouier amplitudes of FT of image_shg

	"""

	n_sample = image_shg.shape[0]

	image_fft = np.fft.fft2(image_shg[0])
	image_fft[0][0] = 0
	image_fft = np.fft.fftshift(image_fft)
	average_fft = np.zeros(image_fft.shape, dtype=complex)

	fft_angle = np.angle(image_fft, deg=True)
	fft_freqs = np.fft.fftfreq(image_fft.size)
	angles = np.unique(fft_angle)
	fourier_spec = np.zeros(angles.shape)
	
	n_bins = fourier_spec.size

	for n in range(n_sample):
		image_fft = np.fft.fft2(image_shg[n])
		image_fft[0][0] = 0
		average_fft += np.fft.fftshift(image_fft) / n_sample	

	for i in range(n_bins):
		indices = np.where(fft_angle == angles[i])
		fourier_spec[i] += np.sum(np.abs(average_fft[indices])) / 360

	#A = np.sqrt(average_fft * fft_angle.size * fft_freqs**2 * (np.cos(fft_angle)**2 + np.sin(fft_angle)**2))

	sdi = np.mean(fourier_spec) / np.max(fourier_spec)

	return angles, fourier_spec, sdi
