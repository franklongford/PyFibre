"""
PyFibre
Preprocessing Library

Created by: Frank Longford
Created on: 18/02/2019
"""
import logging
import numpy as np

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.exposure import rescale_intensity

logger = logging.getLogger(__name__)


def clip_intensities(image, p_intensity=(1, 98)):
    """
    Pre-process image to remove outliers, reduce noise
    and rescale

    Parameters
    ----------
    image:  array_like (float); shape=(n_y, n_x)
        Image to pre-process
    p_intensity: tuple (float); shape=(2,)
        Percentile range for intensity rescaling
        (used to remove outliers)

    Returns
    -------
    image:  array_like (float); shape=(n_y, n_x)
        Pre-processed image
    """

    logger.debug(
        f"Preprocessing images using clipped "
        f"intensity percentages {p_intensity}")
    low, high = np.percentile(image, p_intensity)
    image = rescale_intensity(
        image, in_range=(low, high), out_range=(0.0, 1.0))

    return image


def nl_means(image, p_denoise=(5, 35)):
    """
    Non-local means denoise algorithm using estimate of
    Gaussian noise

    Parameters
    ----------
    image:  array_like (float); shape=(n_y, n_x)
        Image to pre-process
    p_denoise: tuple (float); shape=(2,)
        Parameters for non-linear means denoise algorithm
        (used to remove noise)

    Returns
    -------
    image:  array_like (float); shape=(n_y, n_x)
        Pre-processed image
    """

    sigma = estimate_sigma(image)
    image = denoise_nl_means(
        image, patch_size=p_denoise[0],
        patch_distance=p_denoise[1],
        fast_mode=True, h=1.2 * sigma,
        sigma=sigma, multichannel=False)

    return image
