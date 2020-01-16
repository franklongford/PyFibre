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

from pyfibre.utilities import label_set

from .segment_utilities import segment_check, draw_network

logger = logging.getLogger(__name__)


def binary_to_stack(binary):
    """Create a segment stack from a global binary"""
    label_image = measure.label(binary.astype(int))
    labels = label_set(label_image)

    binary_stack = np.zeros((labels.size,) + binary.shape, dtype=int)

    for index, label in enumerate(labels):
        binary_stack[index] += np.where(label_image == label, 1, 0)

    return np.where(binary_stack, 1, 0)


def segments_to_stack(segments, shape):
    """Convert a list of scikit-image segments to a single binary mask"""
    stack = np.zeros(
        (len(segments),) + shape, dtype=int)

    for index, segment in enumerate(segments):
        minr, minc, maxr, maxc = segment.bbox
        indices = np.mgrid[minr:maxr, minc:maxc]
        binary_image = np.zeros(shape, dtype=int)
        binary_image[(indices[0], indices[1])] += segment.image
        stack[index] += binary_image

    stack = np.where(stack, 1, 0)

    return stack


def stack_to_segments(stack, intensity_image=None, min_size=0, min_frac=0):
    """Convert a binary mask image to a set of scikit-image segment objects"""

    segments = []

    for binary in stack:
        labels = measure.label(binary.astype(np.int))
        segments += [
            segment
            for segment in regionprops(
                labels, intensity_image=intensity_image)
            if segment_check(segment, min_size, min_frac)
        ]

    areas = [segment.filled_area for segment in segments]
    indices = np.argsort(areas)[::-1]
    segments = [segments[i] for i in indices]

    return segments


def segments_to_binary(segments, shape):
    """Convert a list of scikit-image segments to a single binary mask"""
    binary_image = np.zeros(shape, dtype=int)

    for segment in segments:
        minr, minc, maxr, maxc = segment.bbox
        indices = np.mgrid[minr:maxr, minc:maxc]
        binary_image[(indices[0], indices[1])] += segment.image

    binary_image = np.where(binary_image, 1, 0)

    return binary_image


def binary_to_segments(binary, intensity_image=None, min_size=0, min_frac=0):
    """Convert a binary mask image to a set of scikit-image segment objects"""

    if binary.ndim > 2:
        binary = binary.sum(axis=0)

    labels = measure.label(binary.astype(np.int))

    segments = [
        segment
        for segment in regionprops(
            labels, intensity_image=intensity_image)
        if segment_check(segment, min_size, min_frac)
    ]

    areas = [segment.filled_area for segment in segments]
    indices = np.argsort(areas)[::-1]
    segments = [segments[i] for i in indices]

    return segments


def networks_to_binary(networks, shape, area_threshold=200, iterations=9, sigma=None):
    """Return a global binary representing areas of an image containing networks"""

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


def networks_to_segments(networks, image=None, shape=None,
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

    segments = binary_to_segments(binary,
                                  intensity_image=image)

    return segments
