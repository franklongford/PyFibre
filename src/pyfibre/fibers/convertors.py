import logging
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from skimage import draw
from skimage.morphology import remove_small_holes

from pyfibre.tools.convertors import binary_to_regions
from pyfibre.fibers.fibre_utilities import get_node_coord_array

logger = logging.getLogger(__name__)


def draw_network(network, label_image, index=1):
    nodes_coord = get_node_coord_array(network)
    label_image[nodes_coord[:, 0], nodes_coord[:, 1]] = index

    for edge in list(network.edges):
        start = list(network.nodes[edge[1]]["xy"])
        end = list(network.nodes[edge[0]]["xy"])
        line = draw.line(*(start + end))
        label_image[line] = index

    return label_image


def networks_to_binary(networks, shape, area_threshold=200, iterations=9, sigma=None):
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
        smoothed = gaussian_filter(binary.astype(float), sigma=sigma)
        # Convert float image back to binary
        binary = np.where(smoothed, 1, 0)

    # Remove smooth holes with area less than threshold
    binary = remove_small_holes(binary.astype(bool), area_threshold=area_threshold)

    return binary.astype(int)


def networks_to_regions(
    networks, image=None, shape=None, area_threshold=200, iterations=9, sigma=None
):
    """Transform fibre networks into a set of scikit-image segments"""

    # If no intensity image is provided, make sure binary
    # shape is provided
    if image is None:
        assert shape
    else:
        shape = image.shape

    binary = networks_to_binary(
        networks,
        shape,
        area_threshold=area_threshold,
        iterations=iterations,
        sigma=sigma,
    )

    regions = binary_to_regions(binary, intensity_image=image)

    return regions
