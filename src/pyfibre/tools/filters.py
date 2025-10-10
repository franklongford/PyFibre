import numpy as np

from scipy.ndimage.filters import gaussian_filter

from skimage.feature import structure_tensor
from skimage.filters import (
    threshold_li,
    threshold_isodata,
    threshold_mean,
    apply_hysteresis_threshold,
    sato,
)


def gaussian(image, sigma=None):
    """Perform gaussian smoothing on image using sigma
    standard deviation"""

    if sigma is None:
        return image
    return gaussian_filter(image, sigma=sigma)


def tubeness(image, sigma_max=3):
    """Wrapper around the scikit-image sato tubeness filter"""

    tube = sato(
        image,
        sigmas=range(1, sigma_max + 1),
        black_ridges=False,
        mode="constant",
    )

    return tube


def hysteresis(image, alpha=1.0):
    """Hystersis thresholding with low and high clipped values
    determined by the mean, li and isodata threshold"""

    low = np.min([alpha * threshold_mean(image), threshold_li(image)])
    high = threshold_isodata(image)

    threshold = apply_hysteresis_threshold(image, low, high)

    return threshold


def derivatives(image, rank=1):
    """
    Returns derivates of order "rank" for imput image at each pixel

    Parameters
    ----------

    image:  array_like (float); shape(n_y, n_x)
        Image to analyse

    rank:  int (optional)
        Order of derivatives to return (1 = first order, 2 = second order)

    Returns
    -------

    derivative:  array_like (float); shape=(2 or 4, n_y, n_x)
        First or second order derivatives at each image pixel
    """

    derivative = np.zeros(((2,) + image.shape))
    derivative[0] += np.nan_to_num(np.gradient(image, edge_order=1, axis=-2))
    derivative[1] += np.nan_to_num(np.gradient(image, edge_order=1, axis=-1))

    if rank == 2:
        second_derivative = np.zeros(((4,) + image.shape))
        second_derivative[0] += np.nan_to_num(
            np.gradient(derivative[0], edge_order=1, axis=-2)
        )
        second_derivative[1] += np.nan_to_num(
            np.gradient(derivative[1], edge_order=1, axis=-2)
        )
        second_derivative[2] += np.nan_to_num(
            np.gradient(derivative[0], edge_order=1, axis=-1)
        )
        second_derivative[3] += np.nan_to_num(
            np.gradient(derivative[1], edge_order=1, axis=-1)
        )

        return second_derivative

    return derivative


def form_nematic_tensor(image, sigma=0.0001):
    """
    form_nematic_tensor(dx_shg, dy_shg)

    Create local nematic tensor n for each pixel in dx_shg, dy_shg

    Parameters
    ----------
    image:  array_like (float); shape(n_y, n_x)
        Image to analyse
    sigma: float, optional
        Gaussian smoothing standard deviation

    Returns
    -------
    n_vector:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
        Flattened 2x2 nematic vector for each pixel in
        dx_shg, dy_shg (n_xx, n_xy, n_yx, n_yy)

    """

    if image.ndim == 2:
        image = image.reshape((1,) + image.shape)
    nframe = image.shape[0]

    dx_shg, dy_shg = derivatives(image)
    r_xy_2 = dx_shg**2 + dy_shg**2
    indicies = np.where(r_xy_2 > 0)

    nxx = np.zeros(dx_shg.shape)
    nyy = np.zeros(dx_shg.shape)
    nxy = np.zeros(dx_shg.shape)

    nxx[indicies] += dy_shg[indicies] ** 2 / r_xy_2[indicies]
    nyy[indicies] += dx_shg[indicies] ** 2 / r_xy_2[indicies]
    nxy[indicies] -= dx_shg[indicies] * dy_shg[indicies] / r_xy_2[indicies]

    for frame in range(nframe):
        nxx[frame] = gaussian_filter(nxx[frame], sigma=sigma)
        nyy[frame] = gaussian_filter(nyy[frame], sigma=sigma)
        nxy[frame] = gaussian_filter(nxy[frame], sigma=sigma)

    n_tensor = np.stack((nxx, nxy, nxy, nyy), -1).reshape(nxx.shape + (2, 2))

    if nframe == 1:
        n_tensor = n_tensor.reshape(n_tensor.shape[1:])

    return n_tensor


def form_structure_tensor(image, sigma=0.0001):
    """
    form_structure_tensor(image)

    Create local structure tensor n for each pixel in image

    Parameters
    ----------
    image:  array_like (float); shape(n_y, n_x)
        Image to analyse
    sigma: float, optional
        Gaussian smoothing standard deviation

    Returns
    -------
    j_tensor:  array_like (float); shape(nframe, n_y, n_x, 2, 2)
        2x2 structure tensor for each pixel in image stack

    """

    if image.ndim == 2:
        image = image.reshape((1,) + image.shape)
    nframe = image.shape[0]

    jxx = np.zeros(image.shape)
    jxy = np.zeros(image.shape)
    jyy = np.zeros(image.shape)

    for frame in range(nframe):
        jxx[frame], jxy[frame], jyy[frame] = structure_tensor(image[frame], sigma=sigma)

    j_tensor = np.stack((jxx, jxy, jxy, jyy), -1).reshape(jxx.shape + (2, 2))

    if nframe == 1:
        j_tensor = j_tensor.reshape(j_tensor.shape[1:])

    return j_tensor
