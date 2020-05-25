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


def spherical_coords(coords):
    """Transform cartesian coordinates to spherical

    Parameters
    ----------
    coords: array_like, shape=(..., 3)
        Values in 3D cartesian coordinates

    Returns
    -------
    x, y, z: array-like, shape=(...)
        Spherical coordinate system in 3D
    """
    x = np.arcsin(coords[..., 0])
    y = np.arcsin(coords[..., 1])
    z = np.arccos(coords[..., 2])

    return x, y, z


def cluster_mask(centres, intensities, boundary=(0.65, 1.1, 1.40, 0.92)):
    """Create new clusters from results of KMeans.
    Attempts to add regularisation parameters

    Parameters
    ----------
    centres: array-like
        Centres of each cluster identified by K-means filter
    intensities: array-like
        Mean image intensity values for each centre
    boundary: tuple, optional
        Minimum values for the boundary between fibre and cell
        regions.

    Returns
    -------
    clusters: array-like
        Indices of centres in cellular regions
    cost:
        Cost associated with segmentation
    """

    # Normalise centroids
    magnitudes = np.sqrt(np.sum(centres ** 2, axis=-1))
    magnitudes = np.repeat(magnitudes, centres.shape[-1])
    centres = centres / magnitudes.reshape(centres.shape)

    # Convert RGB centroids to spherical coordinates
    x, y, z = spherical_coords(centres)

    # Identify centroids in segmentation mask
    mask = (x <= boundary[0]) * (y <= boundary[1])
    mask *= (z <= boundary[2]) * (intensities <= boundary[3])
    clusters = np.argwhere(mask).flatten()

    # Calculate cost function associated with segmentation, based on
    # distance between boundary
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

    # Extract cluster labels for each pixel and centroids
    # corresponding to each cluster
    labels = clusterer.labels_.reshape(image_shape)
    centres = clusterer.cluster_centers_

    return labels, centres


class BDFilter:
    """Performs segmentation filtering using k-means clustering
    on RGB colour channels.
    Adapted from CurveAlign BDcreationHE routine"""

    def __init__(self, n_runs=2, n_clusters=10, p_intensity=(2, 98),
                 sm_size=5, min_size=20, boundary=(0.65, 1.1, 1.40, 0.92),
                 init_size=10000, reassignment_ratio=0.99,
                 max_no_improvement=15):

        self.n_runs = n_runs
        self.n_clusters = n_clusters
        self.p_intensity = p_intensity
        self.sm_size = sm_size
        self.min_size = min_size
        self.boundary = boundary
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio
        self.max_no_improvement = max_no_improvement

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

    def _cluster_generator(self, rgb_image):
        """Identify pixel clusters in RGB image that correspond
        to cellular regions

        Parameters
        ----------
        rgb_image: array-like, shape=(N, M, 3)
            RGB image to cluster

        Yields
        ------
        labels: array-like, shape=(N, M)
        clusters: array-like
        cost: float
        """

        # Generate greyscale image of RGB
        greyscale = rgb2grey(rgb_image.astype(np.float64))
        greyscale /= greyscale.max()

        for run in range(self.n_runs):

            label_image, centres = cluster_colours(
                rgb_image,
                n_clusters=self.n_clusters,
                init_size=self.init_size,
                reassignment_ratio=self.reassignment_ratio,
                max_no_improvement=self.max_no_improvement
            )

            # Calculate average intensity of each labelled region
            intensities = np.zeros(self.n_clusters)
            for index in range(self.n_clusters):
                cluster_values = greyscale[np.where(label_image == index)]
                intensities[index] += nonzero_mean(cluster_values)

            # Create cluster mask based on centre proximity to boundary
            clusters, cost = cluster_mask(
                centres, intensities, self.boundary)

            yield label_image, clusters, cost

    def filter_image(self, image):

        image_scaled = self._scale_image(image)

        tot_labels = []
        tot_clusters = []
        cost_func = []

        # Perform multiple runs of K-means clustering algorithm
        for results in self._cluster_generator(image_scaled):
            labels, clusters, cost = results

            tot_labels.append(labels)
            tot_clusters.append(clusters)
            cost_func.append(cost)

        # Identify segmentation with lowest cost (best separation)
        min_cost = np.argmin(cost_func)
        labels = tot_labels[min_cost]
        clusters = tot_clusters[min_cost]

        # Create binary stack corresponding to each cluster region
        binary_mask = np.zeros(image.shape[:-1], dtype=bool)
        for cluster in clusters:
            indices = np.where(labels == cluster)
            binary_mask[indices] = True

        # Dilate binary image to smooth regions and remove
        # small holes / objects
        binary_mask = binary_opening(binary_mask, iterations=2)
        binary_mask = binary_fill_holes(binary_mask)
        for _ in range(2):
            binary_mask = remove_small_objects(
                ~binary_mask, min_size=self.min_size)

        return binary_mask
