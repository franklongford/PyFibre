import logging

import networkx as nx
import numpy.typing as npt

from skimage.exposure import equalize_adapthist, rescale_intensity

from pyfibre.tools.preprocessing import clip_intensities, nl_means
from pyfibre.fibers.network_extraction import build_network

logger = logging.getLogger(__name__)


def network_analysis(
    shg_image: npt.NDArray[float],
    p_intensity: tuple[float, float] = (1, 99),
    p_denoise: tuple[float, float] = (5, 35),
    **fire_parameters,
) -> nx.Graph:
    """Perform FIRE algorithm on image and save networkx
    objects for further analysis

    Parameters
    ----------
    shg_image
        SHH image to analyse
    p_intensity
        Percentile range for intensity rescaling
        (used to remove outliers)
    p_denoise
        Parameters for non-linear means denoise algorithm
        (used to remove noise)
    """

    logger.debug("Applying AHE to SHG image")
    norm_image = clip_intensities(shg_image, p_intensity=p_intensity)
    image_equal = equalize_adapthist(rescale_intensity(norm_image))

    logger.debug("Performing NL Denoise using local windows {} {}".format(*p_denoise))
    image_nl = nl_means(image_equal, p_denoise=p_denoise)

    # Call FIRE algorithm to extract full image network
    logger.debug(f"Calling FIRE algorithm using parameter overrides {fire_parameters}")
    return build_network(
        image_nl, **fire_parameters
    )
