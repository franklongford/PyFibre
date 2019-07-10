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

from skimage.feature import greycoprops, greycomatrix
from skimage.measure import shannon_entropy
from scipy.ndimage.filters import gaussian_filter

from pyfibre.model.tools.extraction import branch_angles
from pyfibre.utilities import matrix_split

logger = logging.getLogger(__name__)


def fourier_transform_analysis(image, n_split=1, sigma=None, nbins=200):
    """
    Calculates fourier amplitude spectrum for image

    Parameters
    ----------

    image:  array_like (float); shape=(n_x, n_y)
    Image to analyse

    Returns
    -------

    angles:  array_like (float); shape=(n_bins)
        Angles corresponding to fourier amplitudes

    fourier_spec:  array_like (float); shape=(n_bins)
        Average Fouier amplitudes of FT of image_shg

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
    tensor_analysis(tensor)

    Calculates eigenvalues and eigenvectors of average tensor over area^2 pixels for n_samples

    Parameters
    ----------

    tensor:  array_like (float); shape(nframe, nx, ny, 2, 2)
        Average tensor over area under examination

    Returns
    -------

    tot_anis:  array_like (float); shape=(n_frame, nx, ny)
        Difference between eigenvalues of average tensors

    tot_angle:  array_like (float); shape=(n_frame, nx, ny)
        Angle of dominant eigenvector of average tensors

    tot_energy:  array_like (float); shape=(n_frame, nx, ny)
        Determinent of eigenvalues of average tensors

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


def angle_analysis(angles, weights, N=200):

    angle_hist, _ = np.histogram(angles.flatten(), bins=N,
                                 weights=weights.flatten(),
                                 density=True)
    angle_sdi = angle_hist.mean() / angle_hist.max()

    return angle_sdi


def fibre_analysis(tot_fibres, verbose=False):

    fibre_lengths = np.empty((0,), dtype='float64')
    fibre_waviness = np.empty((0,), dtype='float64')
    fibre_angles = np.empty((0,), dtype='float64')

    for fibre in tot_fibres:

        start = fibre.node_list[0]
        end = fibre.node_list[-1]

        logger.debug(f"N nodes: {len(fibre.node_list)} Length: {fibre.fibre_l}\n"
                     f"Displacement: {fibre.euclid_l}  Direction: {fibre.direction}\n ")

        fibre_lengths = np.concatenate((fibre_lengths, [fibre.fibre_l]))
        fibre_waviness = np.concatenate((fibre_waviness, [fibre.euclid_l / fibre.fibre_l]))

        cos_the = branch_angles(fibre.direction, np.array([[0, 1]]), np.ones(1))
        fibre_angles = np.concatenate((fibre_angles, np.arccos(cos_the) * 180 / np.pi))

    return fibre_lengths, fibre_waviness, fibre_angles


def greycoprops_edit(P, prop='contrast'):
    """Edited version of the scikit-image greycoprops function,
    including additional properties"""
    (num_level, num_level2, num_dist, num_angle) = P.shape

    assert num_level == num_level2
    assert num_dist > 0
    assert num_angle > 0

    if prop in ['similarity', 'mean', 'covariance',
                'cluster', 'entropy', 'autocorrelation']:
        pass
    else:
        return greycoprops(P, prop)

    # normalize each GLCM
    I, J = np.ogrid[0:num_level, 0:num_level]
    P = P.astype(np.float64)
    glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    if prop in ['similarity']:
        weights = 1. / (1. + abs(I - J))
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]

    elif prop == 'autocorrelation':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))

        results = np.apply_over_axes(np.sum, (P * I * J),
                                 axes=(0, 1))[0, 0]

    elif prop == 'mean':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        mean_i = np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        mean_j = np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        results = 0.5 * (mean_i + mean_j)

    elif prop == 'covariance':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        results = np.apply_over_axes(np.sum, (P * (diff_i * diff_j)),
                                 axes=(0, 1))[0, 0]

    elif prop == 'cluster':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        results = np.apply_over_axes(np.sum, (P * (I + J - diff_i - diff_j)),
                                 axes=(0, 1))[0, 0]

    elif prop == 'entropy':
        nat_log = np.log(P)

        mask_0 = P < 1e-15
        mask_0[P < 1e-15] = True
        nat_log[mask_0] = 0

        results = np.apply_over_axes(np.sum, (P * (- nat_log)),
                                 axes=(0, 1))[0, 0]

    return results


def segment_analysis(image, segment, n_tensor, tag):

    database = pd.Series()

    minr, minc, maxr, maxc = segment.bbox
    indices = np.mgrid[minr:maxr, minc:maxc]

    segment_image = image[(indices[0], indices[1])]
    segment_n_tensor = n_tensor[(indices[0], indices[1])]

    (segment_anis_map,
     segment_angle_map,
     segment_angle_map) = tensor_analysis(segment_n_tensor)

    _, _, database[f"{tag} Fourier SDI"] = (0, 0, 0)#fourier_transform_analysis(segment_image)
    database[f"{tag} Angle SDI"] = angle_analysis(segment_angle_map, segment_anis_map)

    database[f"{tag} Mean"] = np.mean(segment_image)
    database[f"{tag} STD"] = np.std(segment_image)
    database[f"{tag} Entropy"] = shannon_entropy(segment_image)

    segment_anis, _ , _ = tensor_analysis(np.mean(segment_n_tensor, axis=(0, 1)))
    database[f"{tag} Anisotropy"] = segment_anis[0]
    database[f"{tag} Pixel Anisotropy"] = np.mean(segment_anis_map)

    database[f"{tag} Area"] = segment.area
    database[f"{tag} Linearity"] = 1 - np.pi * segment.equivalent_diameter / segment.perimeter
    database[f"{tag} Eccentricity"] = segment.eccentricity
    database[f"{tag} Density"] = np.sum(segment_image * segment.image) / segment.area
    database[f"{tag} Coverage"] = segment.extent
    segment_hu = segment.moments_hu

    database[f"{tag} Hu Moment 1"] = segment_hu[0]
    database[f"{tag} Hu Moment 2"] = segment_hu[1]
    database[f"{tag} Hu Moment 3"] = segment_hu[2]
    database[f"{tag} Hu Moment 4"] = segment_hu[3]

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


def fibre_segment_analysis(image_shg, networks, networks_red,
                           fibres, segments, n_tensor):
    """
    Analyse extracted fibre network
    """
    l_regions = len(segments)
    segment_metrics = pd.DataFrame()

    iterator = zip(np.arange(l_regions), networks, networks_red, fibres, segments)

    for i, network, network_red, fibre, segment in iterator:
        # if segment.filled_area >= 1E-2 * image_shg.size:

        segment_series = pd.Series(name=i)

        metrics = segment_analysis(image_shg, segment, n_tensor, 'SHG Fibre')
        segment_series = pd.concat((segment_series, metrics))

        metrics = network_analysis(network, network_red, 'SHG Fibre')
        segment_series = pd.concat((segment_series, metrics))

        fibre_len, fibre_wav, fibre_ang = fibre_analysis(fibre)
        segment_series['SHG Fibre Waviness'] = np.nanmean(fibre_wav)
        segment_series['SHG Fibre Length'] = np.nanmean(fibre_len)
        segment_series['SHG Fibre Cross-Link Density'] = segment_series['SHG Fibre Network Cross-Links'] / len(fibre)

        segment_metrics = segment_metrics.append(segment_series, ignore_index=True)

    # fibre_angle_sdi[i] = angle_analysis(fibre_ang, np.ones(fibre_ang.shape))

    return segment_metrics


def cell_segment_analysis(image, cells, n_tensor, tag='Cell'):

    segment_metrics = pd.DataFrame()

    for i, cell in enumerate(cells):

        segment_series = pd.Series(name=i)

        metrics = segment_analysis(image, cell, n_tensor, f'PL {tag}')
        segment_series = pd.concat((segment_series, metrics))

        segment_metrics = segment_metrics.append(segment_series, ignore_index=True)

    return segment_metrics