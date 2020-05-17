"""
PyFibre
Image Segmentation Library

Created by: Frank Longford
Created on: 18/02/2019

Last Modified: 18/02/2019
"""

import logging

import numpy as np
from scipy.ndimage import (
    gaussian_filter)
from skimage.transform import rescale, resize
from skimage.morphology import remove_small_holes

from .bd_cluster import BDFilter
from .convertors import (
    regions_to_binary, networks_to_regions)
from .utilities import region_swap

logger = logging.getLogger(__name__)


def rgb_segmentation(image_stack, scale=1.0):
    """Return binary filter for cellular identification"""

    if not isinstance(image_stack, np.ndarray):
        image_stack = np.stack(image_stack, axis=-1)

    n_channels = image_stack.shape[-1]
    shape = image_stack.shape[:-1]

    # Create composite RGB image from SHG, PL and transmission
    magnitudes = np.sqrt(np.sum(image_stack**2, axis=-1))
    indices = np.nonzero(magnitudes)
    image_stack[indices] /= np.repeat(
        magnitudes[indices], n_channels).reshape(
        indices[0].shape + (n_channels,))

    # Up-scale image to improve accuracy of clustering
    logger.debug(f"Rescaling by {scale}")
    image_stack = rescale(
        image_stack, scale, multichannel=True,
        mode='constant', anti_aliasing=None
    )

    # Form mask using Kmeans Background filter
    logger.debug(f"Performing BD Filter")
    bd_filter = BDFilter()
    mask_image = bd_filter.filter_image(image_stack)

    # Reducing image to original size
    logger.debug(f"Rescaling image back to {shape}")
    mask_image = resize(
        mask_image, shape, mode='reflect',
        anti_aliasing=True
    )

    # Create cell and fibre global image masks
    cell_mask = np.array(mask_image, dtype=bool)
    fibre_mask = np.where(mask_image, False, True)

    return fibre_mask, cell_mask


def fibre_cell_region_swap(multi_image, fibre_mask, cell_mask):
    """Obtain segments from masks and swap over any incorrectly
    assigned segments"""

    region_swap(
        [cell_mask, fibre_mask],
        [multi_image.pl_image, multi_image.shg_image],
        [250, 150], [0.01, 0.1])

    fibre_mask = remove_small_holes(fibre_mask)
    cell_mask = remove_small_holes(cell_mask)

    return fibre_mask, cell_mask


def create_fibre_filter(fibre_networks, shape,
                        area_threshold=200, iterations=5,
                        sigma=0.5):
    """Create binary filter of fibre regions from a list of
    FibreNetwork instances"""

    graphs = [
        fibre_network.graph
        for fibre_network in fibre_networks
    ]

    regions = networks_to_regions(
        graphs, shape=shape,
        area_threshold=area_threshold,
        iterations=iterations, sigma=sigma)

    # Create a filter for the image that corresponds
    # to the regions that have not been identified as fibrous
    # segments
    fibre_binary = regions_to_binary(
        regions, shape
    )

    # Dilate the binary in order to enhance network
    # regions
    fibre_filter = np.where(fibre_binary, 2, 0.25)
    fibre_filter = gaussian_filter(fibre_filter, 0.5)

    return fibre_filter
