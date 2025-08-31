import logging

import numpy as np
from skimage.transform import rescale, resize


logger = logging.getLogger(__name__)


def normalise_stack(image_stack):
    """Normalise intensity values for each image in stack

    Parameters
    ----------
    image_stack: array-like, shape=(I, J, N)
        Stack of N images

    Returns
    -------
    image_stack: array-like, shape=(I, J, N)
        Stack of images, with intensity values normalised
        across each image
    """

    n_channels = image_stack.shape[-1]

    magnitudes = np.sqrt(np.sum(image_stack**2, axis=-1))
    indices = np.nonzero(magnitudes)

    image_stack[indices] /= np.repeat(magnitudes[indices], n_channels).reshape(
        indices[0].shape + (n_channels,)
    )

    return image_stack


def rgb_segmentation(image_stack, bd_filter, scale=1.0):
    """Return binary filter for cellular identification

    Parameters
    ----------
    image_stack: array-like, shape=(N, I, J)
        Stack of images
    bd_filter: BaseBDFilter
        Instance of filtering algorithm to be used
    scale: float, optional
        Ratio to rescale size of image to

    Returns
    -------
    fibre_mask, cell_mask: array-like, shape=(I, J)
        Binary masks that identify pixels in fibrous and
        cellular regions
    """

    if not isinstance(image_stack, np.ndarray):
        image_stack = np.stack(image_stack, axis=-1)

    shape = image_stack.shape[:-1]

    # Normalise the intensity values of each channel
    image_stack = normalise_stack(image_stack)

    # Up-scale image to improve accuracy of clustering
    logger.debug(f"Rescaling by {scale}")
    image_stack = rescale(
        image_stack, scale, channel_axis=1, mode="constant", anti_aliasing=None
    )

    # Form mask using Kmeans Background filter
    logger.debug("Performing BD Filter")
    mask_image = bd_filter.filter_image(image_stack)

    # Reducing image to original size
    logger.debug(f"Rescaling image back to {shape}")
    mask_image = resize(mask_image, shape, mode="reflect", anti_aliasing=False)

    # Create cell and fibre global image masks
    cell_mask = np.array(mask_image, dtype=bool)
    fibre_mask = np.where(mask_image, False, True)

    return fibre_mask, cell_mask
