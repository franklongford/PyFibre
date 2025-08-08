import logging
import numpy as np

from scipy.ndimage.filters import gaussian_filter

from pyfibre.utilities import matrix_split

logger = logging.getLogger(__name__)


def fourier_transform_analysis(image, n_split=1, sigma=None):
    """
    Calculates fourier amplitude spectrum for image

    Parameters
    ----------
    image :  array_like
        Image to analyse
    n_split : int, optional
        Number of samples to split image into
    sigma : float, optional
        Standard deviation of Gaussian filter to be applied before
        FFT
    n_bins : int, optional
        Number of bins for angular histogram

    Returns
    -------
    angles, fourier_spec:  array_like of floats
        Angles and average fourier amplitudes of FFT performed on
        image
    sdi: float

    """

    if sigma is not None:
        image = gaussian_filter(image, sigma)

    sampled_regions = matrix_split(image, n_split, n_split)
    sdi = []

    for region in sampled_regions:
        image_fft = np.fft.fft2(region)
        image_fft[0][0] = 0
        # image_fft = np.fft.fftshift(image_fft)

        # real = np.real(image_fft)
        # imag = np.imag(image_fft)

        # magnitude = np.abs(image_fft)
        # phase = np.angle(image_fft, deg=True)

        image_grid = np.mgrid[: region.shape[0], : region.shape[1]]
        for i in range(2):
            image_grid[i] -= region.shape[0] * np.array(
                2 * image_grid[i] / region.shape[0], dtype=int
            )
        image_radius = np.sqrt(np.sum(image_grid**2, axis=0))

        angles = image_grid[0] / image_radius
        angles = np.arccos(angles) * 360 / np.pi
        angles[0][0] = 0
        angles = np.fft.fftshift(angles)

        fourier_spec = np.ones(angles.shape)
        sdi.append(np.mean(fourier_spec) / np.max(fourier_spec))

    return angles, fourier_spec, sdi


def tensor_analysis(tensor):
    """
    Calculates eigenvalues and eigenvectors of average tensor over
    area^2 pixels for n_samples

    Parameters
    ----------
    tensor :  array_like of floats
        Average tensor over area under examination. Can either
        refer to a single image or stack of images; in which case outer
        dimension must represent a different image in the stack

    Returns
    -------
    tot_coher, tot_angle, tot_energy : array_like of floats
        Coherence, angle and energy values of input tensor
    """

    if tensor.ndim == 2:
        tensor = tensor.reshape((1,) + tensor.shape)

    eig_val, eig_vec = np.linalg.eigh(tensor)

    eig_diff = np.diff(eig_val, axis=-1).max(axis=-1)
    eig_sum = eig_val.sum(axis=-1)
    indicies = np.nonzero(eig_sum)

    # Calculate the coherence for each tensor
    tot_coher = np.zeros(tensor.shape[:-2])
    tot_coher[indicies] += (eig_diff[indicies] / eig_sum[indicies]) ** 2

    tot_angle = (
        0.5
        * np.arctan2(2 * tensor[..., 1, 0], (tensor[..., 1, 1] - tensor[..., 0, 0]))
        / np.pi
        * 180
    )
    tot_energy = np.trace(np.abs(tensor), axis1=-2, axis2=-1)

    return tot_coher, tot_angle, tot_energy


def angle_analysis(angles, weights=None, n_bin=200):
    """Creates a histogram of values in array `angles`, with optional
    argument `weights`. Returns SDI value of each binned angle in
    histogram.

    Parameters
    ----------
    angles : array_like of floats
        Array of angles to bin
    weights : array_like of floats, optional
        Array of weights corresponding to each value in angles
    n_bin : int, optional
        Number of bins for histogram

    Returns
    -------
    angle_sdi, angle_x: array_like of floats
        Angle and SDI values corresponding to each bin in histogram
    """

    if not isinstance(angles, np.ndarray):
        angles = np.array(angles)

    if weights is None:
        weights = np.ones(angles.shape)

    angle_hist, angle_x = np.histogram(
        angles.flatten(), bins=n_bin, weights=weights.flatten(), density=True
    )
    angle_sdi = angle_hist.mean() / angle_hist.max()

    return angle_sdi, angle_x
