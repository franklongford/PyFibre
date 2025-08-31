import logging

import numpy as np
from scipy.ndimage import gaussian_filter

from pyfibre.fibers.fibre_network import FibreNetwork
from pyfibre.tools.convertors import regions_to_binary, networks_to_regions

logger = logging.getLogger(__name__)


def create_fibre_filter(
    fibre_networks: list[FibreNetwork],
    shape: tuple[..., int],
    area_threshold: int = 200,
    iterations: int = 5,
    sigma: float = 0.5,
):
    """Create binary filter of fibre regions from a list of
    FibreNetwork instances"""

    graphs = [fibre_network.graph for fibre_network in fibre_networks]

    regions = networks_to_regions(
        graphs,
        shape=shape,
        area_threshold=area_threshold,
        iterations=iterations,
        sigma=sigma,
    )

    # Create a filter for the image that corresponds
    # to the regions that have not been identified as fibrous
    # segments
    fibre_binary = regions_to_binary(regions, shape)

    # Dilate the binary in order to enhance network
    # regions
    fibre_filter = np.where(fibre_binary, 2, 0.25)
    fibre_filter = gaussian_filter(fibre_filter, 0.5)

    return fibre_filter
