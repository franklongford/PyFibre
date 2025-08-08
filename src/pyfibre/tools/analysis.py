import logging
import numpy as np

logger = logging.getLogger(__name__)


def tensor_analysis(
    tensor: np.typing.NDArray[float],
) -> tuple[
    np.typing.NDArray[float], np.typing.NDArray[float], np.typing.NDArray[float]
]:
    """
    Calculates eigenvalues and eigenvectors of average tensor over
    area^2 pixels for n_samples

    Parameters
    ----------
    tensor
        Average tensor over area under examination. Can either
        refer to a single image or stack of images; in which case outer
        dimension must represent a different image in the stack

    Returns
    -------
    tot_coher, tot_angle, tot_energy
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


def angle_analysis(
    angles: np.typing.NDArray[float],
    weights: np.typing.NDArray[float] | None = None,
    n_bin: int = 200,
) -> tuple[np.typing.NDArray[float], np.typing.NDArray[float]]:
    """Creates a histogram of values in array `angles`, with optional
    argument `weights`. Returns SDI value of each binned angle in
    histogram.

    Parameters
    ----------
    angles
        Array of angles to bin
    weights
        Array of weights corresponding to each value in angles
    n_bin
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
