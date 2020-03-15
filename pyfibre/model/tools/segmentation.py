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
    gaussian_filter, binary_dilation, binary_closing)
from skimage.transform import rescale, resize
from skimage.morphology import remove_small_holes
from skimage.exposure import equalize_adapthist

from pyfibre.model.objects.segments import CellSegment
from pyfibre.model.tools.segment_utilities import mean_binary

from .bd_cluster import BD_filter
from .convertors import binary_to_regions, regions_to_binary
from .utilities import region_swap

logger = logging.getLogger(__name__)


def rgb_segmentation(
        image_shg, image_pl, image_tran, scale=1.0,
        sigma=0.8, alpha=1.0, min_size=400, edges=False):
    """Return binary filter for cellular identification"""

    min_size *= scale ** 2

    # image_shg = np.sqrt(image_shg * image_tran)
    image_pl = np.sqrt(image_pl * image_tran)
    image_tran = equalize_adapthist(image_tran)

    # Create composite RGB image from SHG, PL and transmission
    image_stack = np.stack((image_shg, image_pl, image_tran), axis=-1)
    magnitudes = np.sqrt(np.sum(image_stack**2, axis=-1))
    indices = np.nonzero(magnitudes)
    image_stack[indices] /= np.repeat(
        magnitudes[indices], 3).reshape(indices[0].shape + (3,))

    # Up-scale image to improve accuracy of clustering
    logger.debug(f"Rescaling by {scale}")
    image_stack = rescale(
        image_stack, scale, multichannel=True,
        mode='constant', anti_aliasing=None
    )

    # Form mask using Kmeans Background filter
    mask_image = BD_filter(image_stack)

    # Reducing image to original size
    logger.debug(
        f"Resizing to {image_shg.shape[0]} x {image_shg.shape[1]} pix")
    mask_image = resize(
        mask_image, image_shg.shape, mode='reflect',
        anti_aliasing=True
    )

    # Create cell and fibre global image masks
    cell_mask = np.array(mask_image, dtype=bool)
    fibre_mask = np.where(mask_image, False, True)

    # Obtain segments from masks and swap over any incorrectly
    # assigned segments
    segment_swap(
        [cell_mask, fibre_mask], [image_pl, image_shg],
        [250, 150], [0.01, 0.1])

    logger.debug("Removing small holes")
    fibre_mask = remove_small_holes(fibre_mask)
    cell_mask = remove_small_holes(cell_mask)

    # Generate lists of scikit-image segments from fibre and
    # cell binaries
    sorted_fibres = binary_to_segments(
        fibre_mask, image_shg, 100, 0.1)
    sorted_cells = binary_to_segments(
        cell_mask, image_pl, 200, 0.001)

    return sorted_cells, sorted_fibres


def cell_segmentation(
        multi_image, fibre_networks, scale=1.0, pl_analysis=False):

    fibre_segments = [
        fibre_network.segment
        for fibre_network in fibre_networks
    ]

    if pl_analysis:

        # Create a filter for the SHG image that enhances the segments
        # identified by the FIRE algorithm
        fibre_net_binary = segments_to_binary(
            fibre_segments, multi_image.shape
        )
        fibre_filter = np.where(fibre_net_binary, 2, 0.25)
        fibre_filter = gaussian_filter(fibre_filter, 0.5)

        # Segment the PL image using k-means clustering
        cell_segments, fibre_col_seg = rgb_segmentation(
            multi_image.shg_image * fibre_filter,
            multi_image.pl_image,
            multi_image.trans_image,
            scale=scale)

        # Create a filter for the SHG image that enhances the segments
        # identified by the k-means clustering algorithm
        fibre_col_binary = segments_to_binary(
            fibre_col_seg,  multi_image.shape
        )
        fibre_col_binary = binary_dilation(fibre_col_binary, iterations=2)
        fibre_col_binary = binary_closing(fibre_col_binary)

        # Generate a filter for the SHG image that combines information
        # from both FIRE and k-means algorithms
        fibre_binary = mean_binary(
            np.array([fibre_net_binary, fibre_col_binary]),
            multi_image.shg_image,
            min_intensity=0.13)

        # Create a new set of segments for each cell region
        cell_segments = binary_to_segments(
            ~fibre_binary, multi_image.pl_image, 250, 0.01)

        cells = [CellSegment(segment=cell_segment,
                             image=multi_image.pl_image)
                 for cell_segment in cell_segments]

    else:
        # Create a filter for the PL image that corresponds
        # to the regions that have not been identified as fibrous
        # segments
        fibre_binary = segments_to_binary(
            fibre_segments, multi_image.shape
        )
        cell_binary = ~fibre_binary

        # Segment the PL image using this filter
        cell_segments = binary_to_segments(
            cell_binary, multi_image.shg_image, 250, 0.01
        )

        cells = [CellSegment(segment=cell_segment,
                             image=multi_image.shg_image)
                 for cell_segment in cell_segments]

    return cells
