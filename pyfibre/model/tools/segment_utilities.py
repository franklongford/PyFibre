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

logger = logging.getLogger(__name__)


def smooth_binary(binary, sigma=None):
    """Smooths binary image based on Guassian filter with
     sigma standard deviation"""

    if sigma is not None:
        smoothed = gaussian_filter(
            binary.astype(float), sigma=sigma
        )
        # Convert float image back to binary
        binary = np.where(smoothed, 1, 0)

    return binary


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
                labels, intensity_image=images[i]):

            if not segment_check(segment_1, min_sizes[i], min_fracs[i]):
                minr, minc, maxr, maxc = segment_1.bbox
                indices = np.mgrid[minr:maxr, minc:maxc]
                masks[i][np.where(labels == segment_1.label)] = False

                segment_2 = measure.regionprops(
                    np.array(segment_1.image, dtype=int),
                    intensity_image=images[j][(indices[0], indices[1])])[0]

                if segment_check(segment_2, 0, min_fracs[j]):
                    masks[j][np.where(labels == segment_1.label)] = True


def mean_binary(binaries, image, iterations=1, min_intensity=0,
                area_threshold=0, sigma=None):
    "Compares two binary of image and produces a filter based on the overlap"

    intensity_map = image * np.mean(binaries, axis=0)

    intensity_mask = np.where(intensity_map > min_intensity, True, False)

    # Remove small holes and objects from masks
    intensity_mask = remove_small_holes(intensity_mask,
                                        area_threshold=area_threshold)
    intensity_mask = remove_small_objects(intensity_mask,
                                          min_size=area_threshold)

    # Dilate image
    binary = binary_dilation(intensity_mask, iterations=iterations)

    smooth_binary(binary, sigma)

    return binary.astype(int)


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
