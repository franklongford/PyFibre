"""
PyFibre
Image Segmentation Library

Created by: Frank Longford
Created on: 18/02/2019

Last Modified: 18/02/2019
"""

import logging
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

from skimage import measure, draw
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops

from pyfibre.utilities import label_set

logger = logging.getLogger(__name__)


def binary_to_stack(binary):

    label_image = measure.label(binary.astype(int))
    labels = label_set(label_image)

    binary_stack = np.zeros((labels.size,) + binary.shape, dtype=int)

    for index, label in enumerate(labels):
        binary_stack[index] += np.where(label_image == label, 1, 0)

    return np.where(binary_stack, 1, 0)


def segment_check(segment, min_size=0, min_frac=0, edges=False, max_x=0, max_y=0):
    """Return whether input segment passes minimum area and average
    intensity checks"""

    segment_check = True
    minr, minc, maxr, maxc = segment.bbox

    if edges:
        edge_check = (minr != 0) * (minc != 0)
        edge_check *= (maxr != max_x)
        edge_check *= (maxc != max_y)

        segment_check *= edge_check

    segment_check *= segment.filled_area >= min_size

    if segment._intensity_image is not None:
        segment_frac = (segment.image * segment.intensity_image).sum() / segment.filled_area
        segment_check *= (segment_frac >= min_frac)

    return segment_check


def segment_swap(masks, images, min_sizes, min_fracs):
    """Performs a segment_check on each segment present in masks using images as
    intensity image. If check fails, removes segment region from mask and performs
    another segment_check using segment of same region with other image as
    intensity image. If this check passes, assigns region onto other mask."""

    for i, j in [[0, 1], [1, 0]]:
        labels = measure.label(masks[i].astype(np.int))
        for segment_1 in measure.regionprops(
                labels, intensity_image=images[i], coordinates='xy'):

            if not segment_check(segment_1, min_sizes[i], min_fracs[i]):
                minr, minc, maxr, maxc = segment_1.bbox
                indices = np.mgrid[minr:maxr, minc:maxc]
                masks[i][np.where(labels == segment_1.label)] = False

                segment_2 = measure.regionprops(
                    np.array(segment_1.image, dtype=int),
                    intensity_image=images[j][(indices[0], indices[1])],
                    coordinates='xy')[0]

                if segment_check(segment_2, 0, min_fracs[j]):
                    masks[j][np.where(labels == segment_1.label)] = True


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
            labels, intensity_image=intensity_image, coordinates='xy')
        if segment_check(segment, min_size, min_frac)
    ]

    areas = [segment.filled_area for segment in segments]
    indices = np.argsort(areas)[::-1]
    segments = [segments[i] for i in indices]

    return segments


def mean_binary(image, binary_1, binary_2, iterations=1, min_size=0, min_intensity=0, thresh=0.6):
    "Compares two segmentations of image and produces a filter based on the overlap"

    image = equalize_adapthist(image)

    intensity_map = 0.5 * image * (binary_1 + binary_2)
    intensity_binary = np.where(intensity_map >= min_intensity, True, False)

    intensity_binary = remove_small_holes(intensity_binary)
    intensity_binary = remove_small_objects(intensity_binary)
    thresholded = binary_dilation(intensity_binary, iterations=iterations)

    smoothed = gaussian_filter(thresholded.astype(np.float), sigma=1.5)
    smoothed = smoothed >= thresh

    return smoothed


def draw_network(network, label_image, index=1):

    nodes_coord = [network.nodes[i]['xy'] for i in network.nodes()]
    nodes_coord = np.stack(nodes_coord)
    label_image[nodes_coord[:,0],nodes_coord[:,1]] = index

    for edge in list(network.edges):
        start = list(network.nodes[edge[1]]['xy'])
        end = list(network.nodes[edge[0]]['xy'])
        line = draw.line(*(start+end))
        label_image[line] = index

    return label_image


def networks_to_binary(networks, image, area_threshold=200, iterations=9, sigma=0.5):

    binary = np.zeros(image.shape, dtype=int)

    "Segment image based on connected components in network"
    for index, network in enumerate(networks):
        draw_network(network, binary, index=1)

    binary = binary_dilation(binary, iterations=iterations)

    if sigma is not None:
        binary = gaussian_filter(
            binary.astype(float), sigma=sigma
        )

    binary = np.rint(binary).astype(bool)

    binary = remove_small_holes(binary, area_threshold=area_threshold)

    return binary.astype(int)


def filter_segments(segments, network, network_red, min_size=200):

    remove_list = []
    for i, segment in enumerate(segments):
        area = np.sum(segment.image)
        if area < min_size: remove_list.append(i)

    for i in remove_list:
        segments.remove(segment[i])
        network.remove(network[i])
        network_red.remove(network_red[i])

    return 	segments, network, network_red
