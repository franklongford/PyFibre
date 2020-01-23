"""
PyFibre
Image Tools Library 

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 18/02/2019
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx

from skimage.feature import greycomatrix
from skimage.measure import shannon_entropy
from scipy.ndimage.filters import gaussian_filter

from pyfibre.utilities import matrix_split
from pyfibre.model.tools.filters import form_structure_tensor

from .feature import greycoprops_edit

logger = logging.getLogger(__name__)


def fourier_transform_analysis(image, n_split=1, sigma=None, n_bins=200):
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
    """

    if sigma != None:
        image = gaussian_filter(image, sigma)

    sampled_regions = matrix_split(image, n_split, n_split)

    for region in sampled_regions:

        image_fft = np.fft.fft2(region)
        image_fft[0][0] = 0
        image_fft = np.fft.fftshift(image_fft)

        real = np.real(image_fft)
        imag = np.imag(image_fft)

        magnitude = np.abs(image_fft)
        phase = np.angle(image_fft, deg=True)

        image_grid = np.mgrid[:region.shape[0], :region.shape[1]]
        for i in range(2):
            image_grid[i] -= region.shape[0] * np.array(2 * image_grid[i] / region.shape[0],
                            dtype=int)
        image_radius = np.sqrt(np.sum(image_grid**2, axis=0))

        angles = image_grid[0] / image_radius
        angles = (np.arccos(angles) * 360 / np.pi)
        angles[0][0] = 0
        angles = np.fft.fftshift(angles)

        sdi = np.mean(fourier_spec) / np.max(fourier_spec)

    return angles, fourier_spec, sdi


def tensor_analysis(tensor):
    """
    Calculates eigenvalues and eigenvectors of average tensor over area^2 pixels
    for n_samples

    Parameters
    ----------
    tensor :  array_like of floats
        Average tensor over area under examination. Can either refer to a single image
        or stack of images; in which case outer dimension must represent a different
        image in the stack

    Returns
    -------
    tot_anis, tot_angle, tot_energy : array_like of floats
        Anisotropy, angle and energy values of input tensor
    """

    if tensor.ndim == 2:
        tensor = tensor.reshape((1,) + tensor.shape)

    eig_val, eig_vec = np.linalg.eigh(tensor)

    eig_diff = np.diff(eig_val, axis=-1).max(axis=-1)
    eig_sum = eig_val.sum(axis=-1)
    indicies = np.nonzero(eig_sum)

    tot_anis = np.zeros(tensor.shape[:-2])
    tot_anis[indicies] += eig_diff[indicies] / eig_sum[indicies]

    tot_angle = 0.5 * np.arctan2(
        2 * tensor[..., 1, 0],
        (tensor[..., 1, 1] - tensor[..., 0, 0])) / np.pi * 180
    tot_energy = np.trace(np.abs(tensor), axis1=-2, axis2=-1)

    return tot_anis, tot_angle, tot_energy


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

    if weights is None:
        weights = np.ones(angles.shape)

    angle_hist, angle_x = np.histogram(
        angles.flatten(), bins=n_bin, weights=weights.flatten(),
        density=True
    )
    angle_sdi = angle_hist.mean() / angle_hist.max()

    return angle_sdi, angle_x


def nematic_tensor_analysis(segment, nematic_tensor, tag=''):
    """Analysis of the nematic tensor"""

    database = pd.Series()

    minr, minc, maxr, maxc = segment.bbox
    indices = np.mgrid[minr:maxr, minc:maxc]
    segment_n_tensor = nematic_tensor[(indices[0], indices[1])]

    (segment_anis_map,
     segment_angle_map,
     segment_angle_map) = tensor_analysis(segment_n_tensor)

    segment_anis, _, _ = tensor_analysis(np.mean(segment_n_tensor, axis=(0, 1)))

    database[f"{tag} Angle SDI"], _ = angle_analysis(segment_angle_map, segment_anis_map)
    database[f"{tag} Anisotropy"] = segment_anis[0]
    database[f"{tag} Pixel Anisotropy"] = np.mean(segment_anis_map)

    return database


def segment_analysis(segment, image=None, tag=''):
    """Analysis for a scikit-image region"""

    database = pd.Series()

    # Perform all non-intensity image relevant metrics
    database[f"{tag} Area"] = segment.area
    database[f"{tag} Linearity"] = 1 - np.pi * segment.equivalent_diameter / segment.perimeter
    database[f"{tag} Eccentricity"] = segment.eccentricity
    database[f"{tag} Coverage"] = segment.extent

    segment_hu = segment.moments_hu
    database[f"{tag} Hu Moment 1"] = segment_hu[0]
    database[f"{tag} Hu Moment 2"] = segment_hu[1]
    database[f"{tag} Hu Moment 3"] = segment_hu[2]
    database[f"{tag} Hu Moment 4"] = segment_hu[3]

    # Check to see whether intensity_image is present or image argument
    # has been supplied
    if image is not None:
        minr, minc, maxr, maxc = segment.bbox
        indices = np.mgrid[minr:maxr, minc:maxc]
        segment_image = image[(indices[0], indices[1])]
    else:
        try:
            segment_image = segment.intensity_image
        except AttributeError:
            return database

    _, _, database[f"{tag} Fourier SDI"] = (0, 0, 0)#fourier_transform_analysis(segment_image)

    database[f"{tag} Mean"] = np.mean(segment_image)
    database[f"{tag} STD"] = np.std(segment_image)
    database[f"{tag} Entropy"] = shannon_entropy(segment_image)
    database[f"{tag} Density"] = np.sum(segment_image * segment.image) / segment.area

    glcm = greycomatrix((segment_image * segment.image * 255.999).astype('uint8'),
                         [1, 2], [0, np.pi/4, np.pi/2, np.pi*3/4], 256,
                         symmetric=True, normed=True)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0

    database[f"{tag} GLCM Contrast"] = greycoprops_edit(glcm, 'contrast').mean()
    database[f"{tag} GLCM Homogeneity"] = greycoprops_edit(glcm, 'homogeneity').mean()
    database[f"{tag} GLCM Energy"] = greycoprops_edit(glcm, 'energy').mean()
    database[f"{tag} GLCM Entropy"] = greycoprops_edit(glcm, 'entropy').mean()
    database[f"{tag} GLCM Autocorrelation"] = greycoprops_edit(glcm, 'autocorrelation').mean()
    database[f"{tag} GLCM Clustering"] = greycoprops_edit(glcm, 'cluster').mean()
    database[f"{tag} GLCM Mean"] = greycoprops_edit(glcm, 'mean').mean()
    database[f"{tag} GLCM Covariance"] = greycoprops_edit(glcm, 'covariance').mean()
    database[f"{tag} GLCM Correlation"] = greycoprops_edit(glcm, 'correlation').mean()

    return database


def network_analysis(network, network_red, tag):

    database = pd.Series()

    cross_links = np.array([degree[1] for degree in network.degree], dtype=int)
    database[f"{tag} Network Cross-Links"] = (cross_links > 2).sum()

    try:
        database[f"{tag} Network Degree"] = nx.degree_pearson_correlation_coefficient(network, weight='r')**2
    except:
        database[f"{tag} Network Degree"] = None

    try:
        database[f"{tag} Network Eigenvalue"] = np.real(nx.adjacency_spectrum(network_red).max())
    except:
        database[f"{tag} Network Eigenvalue"] = None

    try:
        database[f"{tag} Network Connectivity"] = nx.algebraic_connectivity(network_red, weight='r')
    except:
        database[f"{tag} Network Connectivity"] = None

    return database


def fibre_analysis(tot_fibres, nematic_tensor=None):
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

        if nematic_tensor is not None:
            nematic_metrics = nematic_tensor_analysis(
                fibre.segment, nematic_tensor, 'Fibre')
            fibre_series = pd.concat((fibre_series, nematic_metrics))

        database = database.append(fibre_series, ignore_index=True)

    return database


def fibre_network_analysis(fibre_networks, image=None, sigma=0.0001):
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

    if image is not None:
        nematic_tensor = form_structure_tensor(image, sigma)
    else:
        nematic_tensor = form_structure_tensor(
            fibre_networks[0].image, sigma)

    for i, fibre_network in enumerate(fibre_networks):
        # if segment.filled_area >= 1E-2 * image_shg.size:

        fibre_network_series = pd.Series(name=i)

        network_metrics = fibre_network.generate_database()
        fibre_network_series = pd.concat((fibre_network_series, network_metrics))

        fibre_metrics = fibre_analysis(fibre_network.fibres, nematic_tensor)
        mean_fibre_metrics = fibre_metrics.mean()

        fibre_network_series['Mean Fibre Waviness'] = mean_fibre_metrics['Fibre Waviness']
        fibre_network_series['Mean Fibre Length'] = mean_fibre_metrics['Fibre Length']
        fibre_network_series['SHG Fibre Cross-Link Density'] = (
                fibre_network_series['SHG Network Cross-Links'] / len(fibre_metrics))

        database = database.append(fibre_network_series, ignore_index=True)

    # fibre_angle_sdi[i] = angle_analysis(fibre_ang, np.ones(fibre_ang.shape))

    return database


def cell_analysis(cells, image=None, sigma=0.0001):
    """Analysis of a list of `Cell` objects

    Parameters
    ----------
    cells : list of `<class: Cell>`
        List of cells to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from scikit-image regionprops objects
    """
    database = pd.DataFrame()

    if image is not None:
        nematic_tensor = form_structure_tensor(image, sigma)
    else:
        nematic_tensor = form_structure_tensor(
            cells[0].image, sigma)

    for i, cell in enumerate(cells):

        cell_series = segment_analysis(cell.segment, image, 'Cell')

        nematic_metrics = nematic_tensor_analysis(
            cell.segment, nematic_tensor, 'Cell')
        cell_series = pd.concat((cell_series, nematic_metrics))

        database = database.append(cell_series, ignore_index=True)

    return database
