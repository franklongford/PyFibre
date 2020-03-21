"""
PyFibre
BD RGB clustering routine

Created by: Frank Longford
Created on: 20/10/2019
"""

import logging
import numpy as np

from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import (
    binary_fill_holes, binary_opening)

from skimage.util import pad
from skimage.morphology import remove_small_objects
from skimage.color import rgb2grey
from skimage.exposure import equalize_hist

from sklearn.cluster import MiniBatchKMeans

from .preprocessing import clip_intensities

logger = logging.getLogger(__name__)


def nonzero_mean(array):
    """Return mean of non-zero values"""
    return array.sum() / np.count_nonzero(array)


def cluster_mask(centres, intensities, param):
    """Create new clusters from results of KMeans.
    Attempts to add regularisation parameters"""

    # Convert RGB centroids to spherical coordinates
    x = np.arcsin(centres[:, 0])
    y = np.arcsin(centres[:, 1])
    z = np.arccos(centres[:, 2])

    mask = (x <= param[0]) * (y <= param[1])
    mask *= (z <= param[2]) * (intensities <= param[3])

    clusters = np.argwhere(mask).flatten()

    cost = (
        x[clusters].mean() + y[clusters].mean()
        + z[clusters].mean() + intensities[clusters].mean()
    )

    return clusters, cost


def cluster_colours(image, **kwargs):
    """Cluster pixels in an RGB image by their colour using
    Batch KMeans clusterer"""

    image_size = image.shape[0] * image.shape[1]
    image_shape = (image.shape[0], image.shape[1])
    image_channels = image.shape[-1]

    # Perform k-means clustering on PL image
    values = np.array(
        image.reshape((image_size, image_channels)),
        dtype=float)
    clusterer = MiniBatchKMeans(**kwargs)
    clusterer.fit(values)

    labels = clusterer.labels_.reshape(image_shape)
    centres = clusterer.cluster_centers_

    return labels, centres


class BDFilter:
    """Performs segmentation filtering using k-means clustering
    on RGB colour channels.
    Adapted from CurveAlign BDcreationHE routine"""

    def __init__(self, n_runs=2, n_clusters=10, p_intensity=(2, 98),
                 sm_size=5, min_size=20, param=(0.65, 1.1, 1.40, 0.92)):

        self.n_runs = n_runs
        self.n_clusters = n_clusters
        self.p_intensity = p_intensity
        self.sm_size = sm_size
        self.min_size = min_size
        self.param = param

    def _scale_image(self, image):
        """Create a scaled image with enhanced and smoothed RGB
        balance"""

        image_channels = image.shape[-1]
        image_scaled = np.zeros(image.shape, dtype=int)
        pad_size = 10 * self.sm_size

        # Mimic contrast stretching decorrstrech routine in MatLab
        for i in range(image_channels):
            image_scaled[:, :, i] = 255 * clip_intensities(
                image[:, :, i], p_intensity=self.p_intensity)

        # Pad each channel, equalise and smooth to remove
        # salt and pepper noise
        for i in range(image_channels):
            padded = pad(
                image_scaled[:, :, i],
                [pad_size, pad_size],
                'symmetric')
            equalised = 255 * equalize_hist(padded)

            # Double median filter
            for j in range(2):
                equalised = median_filter(
                    equalised, size=(self.sm_size, self.sm_size))

            # Transfer original image from padded back
            image_scaled[:, :, i] = (
                equalised[pad_size: pad_size + image.shape[0],
                          pad_size: pad_size + image.shape[1]])

        return image_scaled

    def _cluster_generator(self, rgb_image, greyscale):

        image_channels = rgb_image.shape[-1]

        for run in range(self.n_runs):

            labels, centres = cluster_colours(
                rgb_image, n_clusters=self.n_clusters,
                init_size=10000, reassignment_ratio=0.99,
                max_no_improvement=15)

            "Reorder labels to represent average intensity"
            intensities = np.zeros(self.n_clusters)

            for index in range(self.n_clusters):
                cluster_values = greyscale[np.where(labels == index)]
                intensities[index] = nonzero_mean(cluster_values)

            magnitudes = np.sqrt(np.sum(centres ** 2, axis=-1))
            magnitudes = np.repeat(magnitudes, image_channels)
            centres = centres / magnitudes.reshape(centres.shape)

            clusters, cost = cluster_mask(
                centres, intensities, self.param)

            yield labels, centres, clusters, cost

    def filter_image(self, image):

        image_scaled = self._scale_image(image)

        # Generate greyscale image of RGB
        greyscale = rgb2grey(image_scaled.astype(np.float64))
        greyscale /= greyscale.max()

        tot_labels = []
        tot_centres = []
        tot_clusters = []
        cost_func = []

        for results in self._cluster_generator(
                image_scaled, greyscale):
            labels, centres, clusters, cost = results

            tot_labels.append(labels)
            tot_centres.append(centres)
            tot_clusters.append(clusters)
            cost_func.append(cost)

        min_cost = np.argmin(cost_func)
        labels = tot_labels[min_cost]
        centres = tot_centres[min_cost]
        clusters = tot_clusters[min_cost]

        logger.info(f"BDFilter centroids: {centres}")

        intensities = np.zeros(self.n_clusters)
        segmented_image = np.zeros(
            (self.n_clusters,) + image.shape,
            dtype=int)
        for label in range(self.n_clusters):
            indices = np.where(labels == label)
            segmented_image[label][indices] += image_scaled[indices]
            intensities[label] = greyscale[indices].sum() / indices[0].size

        # Select blue regions to extract epithelial cells
        epith_cell = np.zeros(image.shape)
        for cluster in clusters:
            epith_cell += segmented_image[cluster]
        epith_grey = rgb2grey(epith_cell)

        # Dilate binary image to smooth regions and remove
        # small holes / objects
        epith_cell_binary = np.where(epith_grey, True, False)
        epith_cell_binary = binary_opening(
            epith_cell_binary, iterations=2)

        binary_x = binary_fill_holes(epith_cell_binary)
        binary_y = remove_small_objects(
            ~binary_x, min_size=self.min_size)

        # Return binary mask for cell identification
        mask_image = remove_small_objects(
            ~binary_y, min_size=self.min_size)

        return mask_image
