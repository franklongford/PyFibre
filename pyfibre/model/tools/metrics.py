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

logger = logging.getLogger(__name__)

NEMATIC_METRICS = ['Angle SDI', 'Anisotropy', 'Local Anisotropy']
SHAPE_METRICS = ['Area', 'Eccentricity', 'Linearity', 'Coverage']
TEXTURE_METRICS = ['Mean', 'STD', 'Entropy']
FIBRE_METRICS = ['Waviness', 'Length']
NETWORK_METRICS = ['Degree', 'Eigenvalue', 'Connectivity',
                   'Cross-Link Density']


def nematic_tensor_metrics(region, nematic_tensor, tag=''):
    """Nematic tensor analysis for a scikit-image region"""

    database = pd.Series(dtype=object)

    minr, minc, maxr, maxc = region.bbox
    indices = np.mgrid[minr:maxr, minc:maxc]
    segment_n_tensor = nematic_tensor[(indices[0], indices[1])]

    (segment_anis_map,
     segment_angle_map,
     segment_angle_map) = tensor_analysis(segment_n_tensor)

    segment_anis, _, _ = tensor_analysis(
        np.mean(segment_n_tensor, axis=(0, 1)))

    database[f"{tag} Angle SDI"], _ = angle_analysis(
        segment_angle_map, segment_anis_map)
    database[f"{tag} Anisotropy"] = segment_anis[0]
    database[f"{tag} Local Anisotropy"] = np.mean(segment_anis_map)

    return database


def region_shape_metrics(region, tag=''):
    """Shape analysis for a scikit-image region"""

    database = pd.Series(dtype=object)

    # Perform all non-intensity image relevant metrics
    database[f"{tag} Area"] = region.area
    ratio = region.equivalent_diameter / region.perimeter
    database[f"{tag} Linearity"] = 1 - np.pi * ratio
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
        minr, minc, maxr, maxc = region.bbox
        indices = np.mgrid[minr:maxr, minc:maxc]
        region_image = image[(indices[0], indices[1])]
    else:
        region_image = region.intensity_image

    # _, _, database[f"{tag} Fourier SDI"] = (0, 0, 0)
    # fourier_transform_analysis(segment_image)

    database[f"{tag} Mean"] = np.mean(region_image)
    database[f"{tag} STD"] = np.std(region_image)
    database[f"{tag} Entropy"] = shannon_entropy(region_image)

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
    except Exception:
        value = None
    database[f"{tag} Network Degree"] = value

    try:
        value = np.real(nx.adjacency_spectrum(network_red).max())
    except Exception:
        value = None
    database[f"{tag} Network Eigenvalue"] = value

    try:
        value = nx.algebraic_connectivity(network_red, weight='r')
    except Exception:
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
        regionrrops objects
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


def segment_metrics(segments, image=None, image_tag=None, sigma=0.0001):
    """Analysis of a list of `BaseSegment` objects

    Parameters
    ----------
    segments : list of `<class: BaseSegment>`
        List of cells to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from scikit-image
        regionprops objects
    """
    database = pd.DataFrame()

    if image is not None:
        nematic_tensor = form_structure_tensor(image, sigma)
    else:
        nematic_tensor = form_structure_tensor(
            segments[0].image, sigma)

    for index, segment in enumerate(segments):

        segment_series = segment.generate_database(
            image_tag=image_tag)

        if image_tag is not None:
            tensor_tag = ' '.join([segment._tag, 'Segment', image_tag])
        else:
            tensor_tag = ' '.join([segment._tag, 'Segment'])

        nematic_metrics = nematic_tensor_metrics(
            segment.region, nematic_tensor,
            tensor_tag)
        segment_series = pd.concat((segment_series, nematic_metrics))

        database = database.append(segment_series, ignore_index=True)

    return database
