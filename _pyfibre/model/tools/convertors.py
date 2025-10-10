"""
PyFibre
Image Segmentation Library

Created by: Frank Longford
Created on: 26/11/2019

Last Modified: 26/11/2019
"""

import logging
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

from skimage import measure
from skimage.morphology import remove_small_holes
from skimage.measure import regionprops

from pyfibre.model.tools.figures import draw_network
from pyfibre.utilities import label_set

from .utilities import region_check, bbox_indices

logger = logging.getLogger(__name__)


def sort_by_area(segments):
    """Sort a list of skimage regionprops segments by
    thier filled_area attribute"""

    areas = [segment.filled_area for segment in segments]
    indices = np.argsort(areas)[::-1]
    sorted_segments = [segments[i] for i in indices]

    return sorted_segments


def binary_to_stack(binary):
    """Create a segment stack from a global binary"""
    label_image = measure.label(binary.astype(int))
    labels = label_set(label_image)

    binary_stack = np.zeros((labels.size,) + binary.shape, dtype=int)

    for index, label in enumerate(labels):
        binary_stack[index] += np.where(label_image == label, 1, 0)

    return np.where(binary_stack, 1, 0)


def stack_to_binary(stack):
    """Converts a stack to a binary image"""

    binary = np.sum(stack, axis=0)
    binary = np.where(binary, 1, 0)

    return binary


def regions_to_stack(regions, shape):
    """Convert a list of scikit-image segments to a single binary mask"""
    stack = np.zeros(
        (len(regions),) + shape, dtype=int)

    for index, region in enumerate(regions):
        binary_image = np.zeros(shape, dtype=int)
        indices = bbox_indices(region)

        binary_image[indices] += region.image
        stack[index] += binary_image

    stack = np.where(stack, 1, 0)

    return stack


def stack_to_regions(stack, intensity_image=None, min_size=0, min_frac=0):
    """Convert a binary mask image to a set of scikit-image
    regionprops objects"""

    regions = []

    for binary in stack:
        labels = measure.label(binary.astype(np.int))
        regions += [
            region
            for region in regionprops(
                labels, intensity_image=intensity_image)
            if region_check(region, min_size, min_frac)
        ]

    regions = sort_by_area(regions)

    return regions


def regions_to_binary(regions, shape):
    """Convert a list of scikit-image segments to a single binary mask"""
    binary_image = np.zeros(shape, dtype=int)

    for region in regions:
        indices = bbox_indices(region)
        binary_image[indices] += region.image

    binary_image = np.where(binary_image, 1, 0)

    return binary_image


def binary_to_regions(binary, intensity_image=None,
                      min_size=0, min_frac=0.0):
    """Convert a binary mask image to a set of scikit-image
    segment objects"""

    if binary.ndim > 2:
        binary = binary.sum(axis=0)

    labels = measure.label(binary.astype(np.int))

    regions = [
        region
        for region in regionprops(
            labels, intensity_image=intensity_image)
        if region_check(region, min_size, min_frac)
    ]

    regions = sort_by_area(regions)

    return regions


def networks_to_binary(networks, shape, area_threshold=200,
                       iterations=9, sigma=None):
    """Return a global binary representing areas of an image
     containing networks"""

    binary = np.zeros(shape, dtype=int)

    # Create skeleton image based on connected components in network
    for index, network in enumerate(networks):
        draw_network(network, binary, index=1)

    # Dilate skeleton image
    if iterations > 0:
        binary = binary_dilation(binary, iterations=iterations)

    # Smooth dilated image
    if sigma is not None:
        smoothed = gaussian_filter(
            binary.astype(float), sigma=sigma
        )
        # Convert float image back to binary
        binary = np.where(smoothed, 1, 0)

    # Remove smooth holes with area less than threshold
    binary = remove_small_holes(
        binary.astype(bool), area_threshold=area_threshold)

    return binary.astype(int)


def networks_to_regions(networks, image=None, shape=None,
                        area_threshold=200, iterations=9,
                        sigma=None):
    """Transform fibre networks into a set of scikit-image segments"""

    # If no intensity image is provided, make sure binary
    # shape is provided
    if image is None:
        assert shape
    else:
        shape = image.shape

    binary = networks_to_binary(networks, shape,
                                area_threshold=area_threshold,
                                iterations=iterations,
                                sigma=sigma)

    regions = binary_to_regions(binary,
                                intensity_image=image)

    return regions


def binary_to_segments(binary, segment_klass,
                       intensity_image=None, min_size=100,
                       min_frac=0.1):
    """Transform binary array into a BaseSegment instance"""

    # Create a new set of segments for each region in binary
    regions = binary_to_regions(
        binary,
        intensity_image=intensity_image,
        min_size=min_size,
        min_frac=min_frac)
    segments = [
        segment_klass(region=region)
        for region in regions
    ]

    return segments


def segments_to_binary(segments, shape):
    """Transform list of BaseSegment instances into a binary array"""

    stack = [
        segment.to_array(shape=shape) for segment in segments
    ]
    binary = stack_to_binary(stack)

    return binary
