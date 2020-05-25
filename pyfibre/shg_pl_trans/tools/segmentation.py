import numpy as np
from scipy.ndimage import binary_dilation, binary_closing
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_mean
from skimage.morphology import remove_small_holes

from pyfibre.model.objects.segments import FibreSegment, CellSegment
from pyfibre.model.tools.convertors import binary_to_segments
from pyfibre.model.tools.segmentation import (
    create_fibre_filter, rgb_segmentation)
from pyfibre.model.tools.utilities import mean_binary, region_swap


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


def shg_segmentation(
        multi_image, fibre_networks,
        min_fibre_size=100, min_cell_size=200,
        min_fibre_frac=100, min_cell_frac=0.001,
        **kwargs):

    fibre_filter = create_fibre_filter(
        fibre_networks, multi_image.shape)

    fibre_binary = np.where(fibre_filter > 0.1, 0, 1)
    cell_binary = np.where(fibre_binary, 0, 1)

    # Create a new set of segments for each fibre region
    fibre_segments = binary_to_segments(
        fibre_binary, FibreSegment,
        intensity_image=multi_image.shg_image,
        min_size=min_fibre_size,
        min_frac=min_fibre_frac)

    # Create a new set of segments for each cell region
    cell_segments = binary_to_segments(
        cell_binary, CellSegment,
        intensity_image=multi_image.shg_image,
        min_size=min_cell_size,
        min_frac=min_cell_frac)

    return fibre_segments, cell_segments


def shg_pl_trans_segmentation(
        multi_image, fibre_networks,
        min_fibre_size=100, min_cell_size=200,
        min_fibre_frac=100, min_cell_frac=0.001,
        scale=1.0, **kwargs):

    # Create an image stack for the rgb_segmentation from SHG and PL
    # images
    fibre_filter = create_fibre_filter(
        fibre_networks, multi_image.shape)

    original_binary = np.where(
        fibre_filter >= threshold_mean(fibre_filter),
        1, 0)

    # Create composite RGB image from SHG, PL and transmission
    stack = (multi_image.shg_image * fibre_filter,
             np.sqrt(multi_image.pl_image * multi_image.trans_image),
             equalize_adapthist(multi_image.trans_image))

    # Segment the PL image using k-means clustering
    fibre_mask, cell_mask = rgb_segmentation(stack, scale=scale)

    # Swap over any pixel areas that may have been wrongly assigned
    fibre_mask, cell_mask = fibre_cell_region_swap(
        multi_image, fibre_mask, cell_mask)

    # Create a binary for the fibrous regions based on information
    # obtained from the rgb segmentation
    fibre_mask = binary_dilation(fibre_mask, iterations=2)
    fibre_mask = binary_closing(fibre_mask)
    new_binary = fibre_mask.astype(int)

    # Generate a filter for the SHG image that combines information
    # from both FIRE and k-means algorithms
    fibre_binary = mean_binary(
        np.array([original_binary, new_binary]),
        multi_image.shg_image,
        min_intensity=0.10)
    cell_binary = np.where(fibre_binary, 0, 1)

    # Create a new set of segments for each fibre region
    fibre_segments = binary_to_segments(
        fibre_binary, FibreSegment,
        intensity_image=multi_image.shg_image,
        min_size=min_fibre_size,
        min_frac=min_fibre_frac)

    # Create a new set of segments for each cell region
    cell_segments = binary_to_segments(
        cell_binary, CellSegment,
        intensity_image=multi_image.shg_image,
        min_size=min_cell_size,
        min_frac=min_cell_frac)

    return fibre_segments, cell_segments
