import logging
import numpy as np

from skimage import measure
from skimage.measure import regionprops

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
        labels = measure.label(binary.astype(np.uint32))
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

    labels = measure.label(binary.astype(np.uint32))

    regions = [
        region
        for region in regionprops(
            labels, intensity_image=intensity_image)
        if region_check(region, min_size, min_frac)
    ]

    regions = sort_by_area(regions)

    return regions
