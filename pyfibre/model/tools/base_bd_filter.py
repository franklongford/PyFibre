"""
PyFibre
BD RGB clustering routine

Created by: Frank Longford
Created on: 20/10/2019
"""

from abc import ABC, abstractmethod
import logging
import numpy as np

from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import (
    binary_fill_holes, binary_opening)

from skimage.color import rgb2grey
from skimage.util import pad
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

from sklearn.cluster import MiniBatchKMeans

from .preprocessing import clip_intensities

logger = logging.getLogger(__name__)


class BaseBDFilter(ABC):
    """Performs segmentation filtering using k-means clustering
    on RGB colour channels.
    Adapted from CurveAlign BDcreationHE routine"""

    def __init__(self, n_runs=2, n_clusters=10, p_intensity=(2, 98),
                 sm_size=5, min_size=20,
                 init_size=10000, reassignment_ratio=0.99,
                 max_no_improvement=15):

        self.n_runs = n_runs
        self.n_clusters = n_clusters
        self.p_intensity = p_intensity
        self.sm_size = sm_size
        self.min_size = min_size
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio
        self.max_no_improvement = max_no_improvement

        self._greyscale = None

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

        # Generate greyscale image of RGB
        self._greyscale = rgb2grey(image_scaled.astype(np.float64))
        self._greyscale /= self._greyscale.max()

        return image_scaled

    def _kmeans_cluster_colours(self, image):
        """Cluster pixels in an RGB image by their colour using
        Batch KMeans clusterer"""

        image_size = image.shape[0] * image.shape[1]
        image_shape = (image.shape[0], image.shape[1])
        image_channels = image.shape[-1]

        # Perform k-means clustering on PL image
        values = np.array(
            image.reshape((image_size, image_channels)),
            dtype=float)
        clusterer = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            init_size=self.init_size,
            reassignment_ratio=self.reassignment_ratio,
            max_no_improvement=self.max_no_improvement)
        clusterer.fit(values)

        # Extract cluster labels for each pixel and centroids
        # corresponding to each cluster
        labels = clusterer.labels_.reshape(image_shape)
        centres = clusterer.cluster_centers_

        return labels, centres

    def _cluster_generator(self, image, **kwargs):
        """Identify pixel clusters in RGB image that correspond
        to cellular regions

        Parameters
        ----------
        image: array-like, shape=(N, M, 3)
            RGB image to cluster

        Yields
        ------
        labels: array-like, shape=(N, M)
        clusters: array-like
        cost: float
        """

        for run in range(self.n_runs):

            label_image, centres = self._kmeans_cluster_colours(
                image)

            mask, cost = self.cellular_classifier(
                label_image, centres,
                **kwargs
            )

            # Create binary stack corresponding to each cluster region
            binary_mask = np.zeros(label_image.shape, dtype=bool)
            for label in np.argwhere(mask):
                indices = np.where(label_image == label)
                binary_mask[indices] = True

            yield binary_mask, cost

    def filter_image(self, image, **kwargs):
        """Performs BD filtering on image"""

        image_scaled = self._scale_image(image)

        tot_masks = []
        cost_func = []

        # Perform multiple runs of K-means clustering algorithm
        for results in self._cluster_generator(
                image_scaled, **kwargs):
            mask, cost = results

            tot_masks.append(mask)
            cost_func.append(cost)

        # Identify segmentation with lowest cost (best separation)
        min_cost = np.argmin(cost_func)
        binary_mask = tot_masks[min_cost]

        # Dilate binary image to smooth regions and remove
        # small holes / objects
        binary_mask = binary_opening(binary_mask, iterations=2)
        binary_mask = binary_fill_holes(binary_mask)
        for _ in range(2):
            binary_mask = remove_small_objects(
                ~binary_mask, min_size=self.min_size)

        self._greyscale = None

        return binary_mask

    @abstractmethod
    def cellular_classifier(self, label_image, centres, **kwargs):
        """Uses labels and centroids generated by K-means clusterer to
        assign each pixel as a cellular region"""
