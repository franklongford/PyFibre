import logging

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix
from skimage.measure import shannon_entropy
from skimage.measure._regionprops import RegionProperties

from pyfibre.tools.analysis import tensor_analysis, angle_analysis
from pyfibre.tools.feature import graycoprops_edit
from pyfibre.tools.filters import form_structure_tensor
from pyfibre.tools.utilities import bbox_sample
from pyfibre.utilities import IMAGE_MAX

logger = logging.getLogger(__name__)

STRUCTURE_METRICS = ["Angle SDI", "Coherence", "Local Coherence"]
SHAPE_METRICS = ["Area", "Eccentricity", "Circularity", "Coverage"]
TEXTURE_METRICS = ["Mean", "STD", "Entropy"]


def _region_sample(
    region: RegionProperties, metric: np.typing.NDArray
) -> np.typing.NDArray:
    """Extract metric values for pixels within segment

    Parameters
    ----------
    region: skimage.RegionProperties
        Region defining pixels within image to analyse
    metric: array-like
        Metric for all pixels in image to be analysed
    """

    # Identify metrics for pixels within bounding box
    metric = bbox_sample(region, metric)

    # Return metrics only for pixels within segment
    indices = np.where(region.image)

    return metric[indices]


def structure_tensor_metrics(
    structure_tensor: np.typing.NDArray, tag: str = ""
) -> pd.Series:
    """Nematic tensor analysis for a scikit-image region"""

    database = pd.Series(dtype=object)

    (segment_coher_map, segment_angle_map, segment_angle_map) = tensor_analysis(
        structure_tensor
    )

    # Calculate mean structure tensor elements
    axis = tuple(range(structure_tensor.ndim - 2))
    mean_tensor = np.mean(structure_tensor, axis=axis)

    segment_coher, _, _ = tensor_analysis(mean_tensor)

    database[f"{tag} Angle SDI"], _ = angle_analysis(
        segment_angle_map, segment_coher_map
    )
    database[f"{tag} Coherence"] = segment_coher[0]
    database[f"{tag} Local Coherence"] = np.mean(segment_coher_map)

    return database


def region_shape_metrics(region: RegionProperties, tag: str = "") -> pd.Series:
    """Shape analysis for a scikit-image region"""

    database = pd.Series(dtype=object)

    # Perform all non-intensity image relevant metrics
    database[f"{tag} Area"] = region.area
    ratio = (np.pi * region.equivalent_diameter) / region.perimeter
    database[f"{tag} Circularity"] = ratio
    database[f"{tag} Eccentricity"] = region.eccentricity
    database[f"{tag} Coverage"] = region.extent

    # segment_hu = region.moments_hu
    # database[f"{tag} Hu Moment 1"] = segment_hu[0]
    # database[f"{tag} Hu Moment 2"] = segment_hu[1]
    # database[f"{tag} Hu Moment 3"] = segment_hu[2]
    # database[f"{tag} Hu Moment 4"] = segment_hu[3]

    return database


def region_texture_metrics(
    region: RegionProperties,
    image: np.typing.NDArray | None = None,
    tag="",
    glcm: bool = False,
) -> pd.Series:
    """Texture analysis for a of scikit-image region"""

    database = pd.Series(dtype=object)

    # Check to see whether intensity_image is present or image argument
    # has been supplied
    if image is not None:
        region_image = bbox_sample(region, image)
    else:
        region_image = region.intensity_image

    # Obtain indices of pixels in region mask
    indices = np.where(region.image)
    intensity_sample = region_image[indices]

    # _, _, database[f"{tag} Fourier SDI"] = (0, 0, 0)
    # fourier_transform_analysis(segment_image)

    database[f"{tag} Mean"] = np.mean(intensity_sample)
    database[f"{tag} STD"] = np.std(intensity_sample)
    database[f"{tag} Entropy"] = shannon_entropy(intensity_sample)

    if glcm:
        glcm = graycomatrix(
            (region_image * region.image * IMAGE_MAX).astype("uint8"),
            [1, 2],
            [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
            256,
            symmetric=True,
            normed=True,
        )
        glcm[0, :, :, :] = 0
        glcm[:, 0, :, :] = 0

        greycoprops = graycoprops_edit(glcm)

        metrics = [
            "Contrast",
            "Homogeneity",
            "Energy",
            "Entropy",
            "Autocorrelation",
            "Clustering",
            "Mean",
            "Covariance",
            "Correlation",
        ]

        for metric in metrics:
            value = greycoprops[metric.lower()].mean()
            database[f"{tag} GLCM {metric}"] = value

    return database


def segment_metrics(
    segments,
    image: np.typing.NDArray,
    image_tag: str | None = None,
    sigma: float = 0.0001,
):
    """Analysis of a list of `Segment` objects

    Parameters
    ----------
    segments : list of `<class: Segment>`
        List of segments to analyse
    image: array-like
        Full image to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from scikit-image
        regionprops objects
    """
    database = pd.DataFrame()

    structure_tensor = form_structure_tensor(image, sigma)

    for index, segment in enumerate(segments):
        if image_tag is not None:
            segment_tag = " ".join([segment.name, "Segment", image_tag])
        else:
            segment_tag = " ".join([segment.name, "Segment"])

        segment_series = pd.Series(dtype=object)

        shape_metrics = region_shape_metrics(segment.region, tag=segment_tag)
        texture_metrics = region_texture_metrics(segment.region, tag=segment_tag)

        segment_series = pd.concat((segment_series, shape_metrics), ignore_index=False)
        segment_series = pd.concat(
            (segment_series, texture_metrics), ignore_index=False
        )

        # Only use pixel tensors in segment
        segment_tensor = _region_sample(segment.region, structure_tensor)

        nematic_metrics = structure_tensor_metrics(segment_tensor, tag=segment_tag)

        segment_series = pd.concat((segment_series, nematic_metrics))

        database = pd.concat((database, segment_series.to_frame().T), ignore_index=True)

    return database
