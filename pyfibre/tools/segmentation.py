"""
PyFibre
Image Segmentation Library 

Created by: Frank Longford
Created on: 18/02/2019

Last Modified: 18/02/2019
"""

import logging
import numpy as np

from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_opening

from skimage import measure, draw
from skimage.util import pad
from skimage.transform import rescale, resize
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.color import rgb2grey
from skimage.exposure import equalize_hist, equalize_adapthist

from sklearn.cluster import MiniBatchKMeans

from pyfibre.model.preprocessing import clip_intensities

logger = logging.getLogger(__name__)


def create_binary_image(segments, shape):

	binary_image = np.zeros(shape)

	for segment in segments:
		minr, minc, maxr, maxc = segment.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]
		binary_image[(indices[0], indices[1])] += segment.image

	binary_image = np.where(binary_image, 1, 0)

	return binary_image


def segment_check(segment, min_size=0, min_frac=0, edges=False, max_x=0, max_y=0):

	segment_check = True
	minr, minc, maxr, maxc = segment.bbox

	if edges:
		edge_check = (minr != 0) * (minc != 0)
		edge_check *= (maxr != max_x)
		edge_check *= (maxc != max_y)

		segment_check *= edge_check

	segment_check *= segment.filled_area >= min_size
	segment_frac = (segment.image * segment.intensity_image).sum() / segment.filled_area
	segment_check *= (segment_frac >= min_frac)

	return segment_check

def get_segments(image, binary, min_size=0, min_frac=0):

	labels = measure.label(binary.astype(np.int))
	segments = []
	areas = []

	for segment in measure.regionprops(labels, intensity_image=image, coordinates='xy'):
		seg_check = segment_check(segment, min_size, min_frac)

		if seg_check:
			segments.append(segment)
			areas.append(segment.filled_area)

	indices = np.argsort(areas)[::-1]
	sorted_segs = [segments[i] for i in indices]

	return sorted_segs


def prepare_composite_image(image, p_intensity=(2, 98), sm_size=7):

	image_size = image.shape[0] * image.shape[1]
	image_shape = (image.shape[0], image.shape[1])
	image_channels = image.shape[-1]
	image_scaled = np.zeros(image.shape, dtype=int)
	pad_size = 10 * sm_size

	"Mimic contrast stretching decorrstrech routine in MatLab"
	for i in range(image_channels):
		image_scaled[:, :, i] = 255 * clip_intensities(image[:, :, i], p_intensity=p_intensity)

	"Pad each channel, equalise and smooth to remove salt and pepper noise"
	for i in range(image_channels):
		padded = pad(image_scaled[:, :, i], [pad_size, pad_size], 'symmetric')
		equalised = 255 * equalize_hist(padded)
		smoothed = median_filter(equalised, size=(sm_size, sm_size))
		smoothed = median_filter(smoothed, size=(sm_size, sm_size))
		image_scaled[:, :, i] = smoothed[pad_size : pad_size + image.shape[0],
						 	pad_size : pad_size + image.shape[1]]

	return image_scaled


def cluster_colours(image, n_clusters=8, n_init=10):

	image_size = image.shape[0] * image.shape[1]
	image_shape = (image.shape[0], image.shape[1])
	image_channels = image.shape[-1]

	"Perform k-means clustering on PL image"
	X = np.array(image.reshape((image_size, image_channels)), dtype=float)
	clustering = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init, 
				    reassignment_ratio=0.99, init_size=n_init*100,
				    max_no_improvement=15)
	clustering.fit(X)

	labels = clustering.labels_.reshape(image_shape)
	centres = clustering.cluster_centers_

	return labels, centres


def BD_filter(image, n_runs=2, n_clusters=10, p_intensity=(2, 98), sm_size=5, param=[0.65, 1.1, 1.40, 0.92]):
	"Adapted from CurveAlign BDcreationHE routine"

	assert image.ndim == 3

	image_size = image.shape[0] * image.shape[1]
	image_shape = (image.shape[0], image.shape[1])
	image_channels = image.shape[-1]

	image_scaled = prepare_composite_image(image, p_intensity, sm_size)

	logger.debug("Making greyscale")
	greyscale = rgb2grey(image_scaled.astype(np.float64))
	greyscale /= greyscale.max()

	tot_labels = []
	tot_centres = []
	tot_cell_clusters = []
	cost_func = np.zeros(n_runs)

	for run in range(n_runs):

		labels, centres = cluster_colours(image_scaled, n_clusters)
		tot_labels.append(labels)

		"Reorder labels to represent average intensity"
		intensities = np.zeros(n_clusters)

		for i in range(n_clusters):
			intensities[i] = greyscale[np.where(labels == i)].sum() / np.where(labels == i, 1, 0).sum()

		magnitudes = np.sqrt(np.sum(centres**2, axis=-1))
		norm_centres = centres / np.repeat(magnitudes, image_channels).reshape(centres.shape)
		tot_centres.append(norm_centres)

		"Convert RGB centroids to spherical coordinates"
		X = np.arcsin(norm_centres[:, 0])
		Y = np.arcsin(norm_centres[:, 1])
		Z = np.arccos(norm_centres[:, 2])
		I = intensities

		"Define the plane of division between cellular and fibourus clusters"
		#data = np.stack((X, Y, Z, I), axis=1)
		#clusterer = KMeans(n_clusters=2)
		#clusterer.fit(data)
		#cell_clusters = clusterer.labels_
			
		cell_clusters = (X <= param[0]) * (Y <= param[1]) * (Z <= param[2]) * (I <= param[3])
		chosen_clusters = np.argwhere(cell_clusters).flatten()
		cost_func[run] += X[chosen_clusters].mean() +  Y[chosen_clusters].mean() \
				 + Z[chosen_clusters].mean() + I[chosen_clusters].mean()
		tot_cell_clusters.append(chosen_clusters)

	labels = tot_labels[cost_func.argmin()]
	norm_centres = tot_centres[cost_func.argmin()]
	cell_clusters = tot_cell_clusters[cost_func.argmin()]

	intensities = np.zeros(n_clusters)
	segmented_image = np.zeros((n_clusters,) + image.shape, dtype=int)
	for i in range(n_clusters):
		segmented_image[i][np.where(labels == i)] += image_scaled[np.where(labels == i)]
		intensities[i] = greyscale[np.where(labels == i)].sum() / np.where(labels == i, 1, 0).sum()

	"Select blue regions to extract epithelial cells"
	epith_cell = np.zeros(image.shape)
	for i in cell_clusters: epith_cell += segmented_image[i]
	epith_grey = rgb2grey(epith_cell)

	"Convert RGB centroids to spherical coordinates"
	X = np.arcsin(norm_centres[:, 0])
	Y = np.arcsin(norm_centres[:, 1])
	Z = np.arccos(norm_centres[:, 2])
	I = intensities

	"""
	print(X, Y, Z, I)
	print((X <= param[0]) * (Y <= param[1]) * (Z <= param[2]) * (I <= param[3]))

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	plt.figure(100, figsize=(10, 10))
	plt.imshow(image)
	plt.axis('off')

	plt.figure(1000, figsize=(10, 10))
	plt.imshow(image_scaled)
	plt.axis('off')

	for i in range(n_clusters):
		plt.figure(i)
		plt.imshow(segmented_image[i])

	not_clusters = [i for i in range(n_clusters) if i not in cell_clusters]

	plt.figure(1001)
	plt.scatter(X[cell_clusters], Y[cell_clusters])
	plt.scatter(X[not_clusters], Y[not_clusters])
	for i in range(n_clusters): plt.annotate(i, (X[i], Y[i]))

	plt.figure(1002)
	plt.scatter(X[cell_clusters], Z[cell_clusters])
	plt.scatter(X[not_clusters], Z[not_clusters])
	for i in range(n_clusters): plt.annotate(i, (X[i], Z[i]))

	plt.figure(1003)
	plt.scatter(X[cell_clusters], I[cell_clusters])
	plt.scatter(X[not_clusters], I[not_clusters])
	for i in range(n_clusters): plt.annotate(i, (X[i], I[i]))

	plt.show()	
	#"""

	"Dilate binary image to smooth regions and remove small holes / objects"
	epith_cell_BW = np.where(epith_grey, True, False)
	epith_cell_BW_open = binary_opening(epith_cell_BW, iterations=2)

	BWx = binary_fill_holes(epith_cell_BW_open)
	BWy = remove_small_objects(~BWx, min_size=20)

	"Return binary mask for cell identification"
	mask_image = remove_small_objects(~BWy, min_size=20)

	return mask_image



def cell_segmentation(image_shg, image_pl, image_tran, scale=1.0, sigma=0.8, alpha=1.0,
			min_size=400, edges=False):
	"Return binary filter for cellular identification"


	min_size *= scale**2

	#image_shg = np.sqrt(image_shg * image_tran)
	image_pl = np.sqrt(image_pl * image_tran)
	image_tran = equalize_adapthist(image_tran)

	"Create composite RGB image from SHG, PL and transmission"
	image_stack = np.stack((image_shg, image_pl, image_tran), axis=-1)
	magnitudes = np.sqrt(np.sum(image_stack**2, axis=-1))
	indices = np.nonzero(magnitudes)
	image_stack[indices] /= np.repeat(magnitudes[indices], 3).reshape(indices[0].shape + (3,))

	"Up-scale image to impove accuracy of clustering"
	logger.debug(f"Rescaling by {scale}")
	image_stack = rescale(image_stack, scale, multichannel=True, mode='constant', anti_aliasing=None)

	"Form mask using Kmeans Background filter"
	mask_image = BD_filter(image_stack)
	logger.debug(f"Resizing to {image_shg[0]} x {image_shg[1]} pix")
	mask_image = resize(mask_image, image_shg.shape, mode='reflect', anti_aliasing=True)

	cells = []
	cell_areas = []
	fibres = []
	fibre_areas = []

	cell_binary = np.array(mask_image, dtype=bool)
	fibre_binary = np.where(mask_image, False, True)

	cell_labels = measure.label(cell_binary.astype(np.int))
	for cell in measure.regionprops(cell_labels, intensity_image=image_pl, coordinates='xy'):
		cell_check = segment_check(cell, 250, 0.01)

		if not cell_check:
			minr, minc, maxr, maxc = cell.bbox
			indices = np.mgrid[minr:maxr, minc:maxc]
			cell_binary[np.where(cell_labels == cell.label)] = False

			fibre = measure.regionprops(np.array(cell.image, dtype=int),
							intensity_image=image_shg[(indices[0], indices[1])],
							coordinates='xy')[0]

			fibre_check = segment_check(fibre, 0, 0.1)
			if fibre_check: fibre_binary[np.where(cell_labels == cell.label)] = True

	fibre_labels = measure.label(fibre_binary.astype(np.int))
	for fibre in measure.regionprops(fibre_labels, intensity_image=image_shg, coordinates='xy'):
		fibre_check = segment_check(fibre, 150, 0.1)

		if not fibre_check:
			minr, minc, maxr, maxc = fibre.bbox
			indices = np.mgrid[minr:maxr, minc:maxc]
			fibre_binary[np.where(fibre_labels == fibre.label)] = False

			cell = measure.regionprops(np.array(fibre.image, dtype=int),
							intensity_image=image_pl[(indices[0], indices[1])],
							coordinates='xy')[0]

			cell_check = segment_check(cell, 0, 0.01)
			if cell_check: cell_binary[np.where(fibre_labels == fibre.label)] = True

	logger.debug("Removing small holes")
	fibre_binary = remove_small_holes(fibre_binary)
	cell_binary = remove_small_holes(cell_binary)

	sorted_fibres = get_segments(image_shg, fibre_binary, 150, 0.1)
	sorted_cells = get_segments(image_pl, cell_binary, 250, 0.01)

	return sorted_cells, sorted_fibres


def mean_binary(image, binary_1, binary_2, iterations=1, min_size=0, min_intensity=0, thresh=0.6):
	"Compares two segmentations of image and produces a filter based on the overlap"

	image = equalize_adapthist(image)

	intensity_map = 0.5 * image * (binary_1 + binary_2)
	intensity_binary = np.where(intensity_map >= min_intensity, True, False)

	intensity_binary = remove_small_holes(intensity_binary)
	intensity_binary = remove_small_objects(intensity_binary)
	thresholded = binary_dilation(intensity_binary, iterations=iterations)
	
	smoothed = gaussian_filter(thresholded.astype(np.float), sigma=1.5)
	smoothed = np.where(smoothed >= thresh, True, False)

	"""
	import matplotlib.pyplot as plt
	plt.figure(0)
	plt.imshow(intensity_map)
	plt.figure(1)
	plt.imshow(thresholded)
	plt.figure(2)
	plt.imshow(smoothed)
	plt.show()
	#"""

	return smoothed


def fibre_segmentation(image_shg, networks, networks_red, area_threshold=200, iterations=9):

	n_net = len(networks)
	fibres = []

	iterator = zip(np.arange(n_net), networks, networks_red)

	"Segment image based on connected components in network"
	for i, network, network_red in iterator:

		label_image = np.zeros(image_shg.shape, dtype=int)
		label_image = draw_network(network, label_image, 1)

		dilated_image = binary_dilation(label_image, iterations=iterations)
		smoothed_image = gaussian_filter(dilated_image, sigma=0.5)
		filled_image = remove_small_holes(smoothed_image, area_threshold=area_threshold)
		binary_image = np.where(filled_image > 0, 1, 0)

		segment = measure.regionprops(binary_image, intensity_image=image_shg, coordinates='xy')[0]
		area = np.sum(segment.image)

		minr, minc, maxr, maxc = segment.bbox
		indices = np.mgrid[minr:maxr, minc:maxc]

		#print(network.number_of_nodes(), area, 1E-2 * image_shg.size)

		segment.label = (i + 1)
		fibres.append(segment)

	return fibres


def draw_network(network, label_image, index):

	nodes_coord = [network.nodes[i]['xy'] for i in network.nodes()]
	nodes_coord = np.stack(nodes_coord)
	label_image[nodes_coord[:,0],nodes_coord[:,1]] = index

	for edge in list(network.edges):
		start = list(network.nodes[edge[1]]['xy'])
		end = list(network.nodes[edge[0]]['xy'])
		line = draw.line(*(start+end))
		label_image[line] = index

	return label_image


def filter_segments(segments, network, network_red, min_size=200):

	remove_list = []
	for i, segment in enumerate(segments):
		area = np.sum(segment.image)
		if area < min_size: remove_list.append(i)
			
	for i in remove_list:
		segments.remove(segment[i])
		network.remove(network[i])
		network_red.remove(network_red[i])
	
	return 	segments, network, network_red

