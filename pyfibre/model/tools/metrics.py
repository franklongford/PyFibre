import logging

import networkx as nx
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix
from skimage.measure import shannon_entropy

from pyfibre.model.tools.analysis import (
    tensor_analysis, angle_analysis)
from pyfibre.model.tools.feature import greycoprops_edit
from pyfibre.model.tools.filters import form_structure_tensor
from pyfibre.model.tools.utilities import bbox_sample

logger = logging.getLogger(__name__)

STRUCTURE_METRICS = ['Angle SDI', 'Coherence', 'Local Coherence']
SHAPE_METRICS = ['Area', 'Eccentricity', 'Circularity', 'Coverage']
TEXTURE_METRICS = ['Mean', 'STD', 'Entropy']
FIBRE_METRICS = ['Waviness', 'Length']
NETWORK_METRICS = ['Degree', 'Eigenvalue', 'Connectivity',
                   'Cross-Link Density']


def _region_sample(region, metric):
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


def structure_tensor_metrics(structure_tensor, tag=''):
    """Nematic tensor analysis for a scikit-image region"""

    database = pd.Series(dtype=object)

    (segment_coher_map,
     segment_angle_map,
     segment_angle_map) = tensor_analysis(structure_tensor)

    # Calculate mean structure tensor elements
    axis = tuple(range(structure_tensor.ndim - 2))
    mean_tensor = np.mean(structure_tensor, axis=axis)

    segment_coher, _, _ = tensor_analysis(mean_tensor)

    database[f"{tag} Angle SDI"], _ = angle_analysis(
        segment_angle_map, segment_coher_map)
    database[f"{tag} Coherence"] = segment_coher[0]
    database[f"{tag} Local Coherence"] = np.mean(segment_coher_map)

    return database


def region_shape_metrics(region, tag=''):
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


def region_texture_metrics(region, image=None, tag='', glcm=False):
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

        glcm = greycomatrix(
            (region_image * region.image * 255.999).astype('uint8'),
            [1, 2], [0, np.pi/4, np.pi/2, np.pi*3/4], 256,
            symmetric=True, normed=True)
        glcm[0, :, :, :] = 0
        glcm[:, 0, :, :] = 0

        greycoprops = greycoprops_edit(glcm)

        metrics = ["Contrast", "Homogeneity", "Energy",
                   "Entropy", "Autocorrelation", "Clustering",
                   "Mean", "Covariance", "Correlation"]

        for metric in metrics:
            value = greycoprops[metric.lower()].mean()
            database[f"{tag} GLCM {metric}"] = value

    return database


def network_metrics(network, network_red, n_fibres, tag=''):
    """Analyse networkx Graph object"""

    database = pd.Series(dtype=object)

    database['No. Fibres'] = n_fibres

    cross_links = np.array(
        [degree[1] for degree in network.degree],
        dtype=int)
    database[f"{tag} Network Cross-Link Density"] = (
        (cross_links > 2).sum() / n_fibres)

    try:
        value = nx.degree_pearson_correlation_coefficient(
            network, weight='r') ** 2
    except Exception as err:
        logger.debug(f'Network Degree calculation failed: {str(err)}')
        value = None
    database[f"{tag} Network Degree"] = value

    try:
        value = np.real(nx.adjacency_spectrum(network_red).max())
    except Exception as err:
        logger.debug(f'Network Eigenvalue calculation failed: {str(err)}')
        value = None
    database[f"{tag} Network Eigenvalue"] = value

    try:
        value = nx.algebraic_connectivity(network_red, weight='r')
    except Exception as err:
        logger.debug(f'Network Connectivity calculation failed: {str(err)}')
        value = None
    database[f"{tag} Network Connectivity"] = value

    return database


def fibre_metrics(tot_fibres):
    """Analysis of list of `Fibre` objects

    Parameters
    ----------
    tot_fibres : list of `<class: Fibre>`
        List of fibre to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from networkx Graph and scikit-image
        regionprops objects
    """

    database = pd.DataFrame()

    for fibre in tot_fibres:
        fibre_series = fibre.generate_database()
        database = database.append(
            fibre_series, ignore_index=True)

    return database


def fibre_network_metrics(fibre_networks):
    """Analysis of list of `FibreNetwork` objects

    Parameters
    ----------
    fibre_networks : list of `<class: FibreNetwork>`
        List of fibre networks to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from networkx Graph and scikit-image
        regionprops objects
    """

    database = pd.DataFrame()

    for i, fibre_network in enumerate(fibre_networks):
        # if segment.filled_area >= 1E-2 * image_shg.size:

        fibre_network_series = pd.Series(name=i, dtype=object)

        metrics = fibre_network.generate_database()

        fibre_network_series = pd.concat(
            (fibre_network_series, metrics))

        database = database.append(
            fibre_network_series, ignore_index=True)

    return database


def segment_metrics(segments, image, image_tag=None, sigma=0.0001):
    """Analysis of a list of `BaseSegment` objects

    Parameters
    ----------
    segments : list of `<class: BaseSegment>`
        List of cells to analyse
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

        segment_series = segment.generate_database(
            image_tag=image_tag)

        if image_tag is not None:
            tensor_tag = ' '.join([segment.tag, 'Segment', image_tag])
        else:
            tensor_tag = ' '.join([segment.tag, 'Segment'])

        # Only use pixel tensors in segment
        segment_tensor = _region_sample(
            segment.region, structure_tensor)

        nematic_metrics = structure_tensor_metrics(
            segment_tensor, tensor_tag)

        segment_series = pd.concat((segment_series, nematic_metrics))

        database = database.append(segment_series, ignore_index=True)

    return database
