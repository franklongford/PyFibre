"""
PyFibre
Image Segmentation Library 

Created by: Frank Longford
Created on: 18/02/2019

Last Modified: 18/02/2019
"""

import logging
import numpy as np

from skimage.transform import rescale, resize
from skimage.morphology import remove_small_holes
from skimage.exposure import equalize_adapthist

from .bd_cluster import BD_filter
from .segment_utilities import segment_swap
from pyfibre.model.tools.convertors import binary_to_segments

logger = logging.getLogger(__name__)


def rgb_segmentation(image_shg, image_pl, image_tran, scale=1.0, sigma=0.8, alpha=1.0,
                     min_size=400, edges=False):
    """Return binary filter for cellular identification"""

    min_size *= scale**2

    #image_shg = np.sqrt(image_shg * image_tran)
    image_pl = np.sqrt(image_pl * image_tran)
    image_tran = equalize_adapthist(image_tran)

    # Create composite RGB image from SHG, PL and transmission
    image_stack = np.stack((image_shg, image_pl, image_tran), axis=-1)
    magnitudes = np.sqrt(np.sum(image_stack**2, axis=-1))
    indices = np.nonzero(magnitudes)
    image_stack[indices] /= np.repeat(magnitudes[indices], 3).reshape(indices[0].shape + (3,))

    # Up-scale image to improve accuracy of clustering
    logger.debug(f"Rescaling by {scale}")
    image_stack = rescale(
        image_stack, scale, multichannel=True, mode='constant', anti_aliasing=None
    )

    # Form mask using Kmeans Background filter
    mask_image = BD_filter(image_stack)

    # Reducing image to original size
    logger.debug(f"Resizing to {image_shg.shape[0]} x {image_shg.shape[1]} pix")
    mask_image = resize(
        mask_image, image_shg.shape, mode='reflect', anti_aliasing=True
    )

    # Create cell and fibre global image masks
    cell_mask = np.array(mask_image, dtype=bool)
    fibre_mask = np.where(mask_image, False, True)

    # Obtain segments from masks and swap over any incorrectly assigned segments
    segment_swap([cell_mask, fibre_mask], [image_pl, image_shg], [250, 150], [0.01, 0.1])

    logger.debug("Removing small holes")
    fibre_mask = remove_small_holes(fibre_mask)
    cell_mask = remove_small_holes(cell_mask)

    # Generate lists of scikit-image segments from fibre and cell binaries
    sorted_fibres = binary_to_segments(image_shg, fibre_mask.astype(int), 150, 0.1)
    sorted_cells = binary_to_segments(image_pl, cell_mask.astype(int), 250, 0.01)

    return sorted_cells, sorted_fibres


