"""
PyFibre
Image Tools Library

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 18/02/2019
"""

from collections import defaultdict

import numpy as np

from skimage.feature.texture import check_nD


def calculate_metric(P, weights):

    results = np.apply_over_axes(
        np.sum, (P * weights), axes=(0, 1))[0, 0]

    return results


def glcm_product_props(P):
    """Calculate properties with weights involving
    product terms"""

    asm = calculate_metric(P, P)

    return asm, np.sqrt(asm)


def glcm_log_props(P):
    """Calculate properties with weights involving
    logarithmic terms"""

    nat_log = np.log(P)
    nat_log[P < 1e-15] = 0
    entropy = calculate_metric(P, -nat_log)

    return entropy


def glcm_difference_props(P):
    """Calculate properties with weights involving
    difference terms"""

    num_level = P.shape[0]
    I, J = np.ogrid[0:num_level, 0:num_level]

    # Create weights for specified property
    diff = I - J
    square = (diff ** 2).reshape((num_level, num_level, 1, 1))
    absolute = (np.abs(diff)).reshape((num_level, num_level, 1, 1))

    contrast = calculate_metric(P, square)
    dissimilarity = calculate_metric(P, absolute)
    homogeneity = calculate_metric(P, 1. / (1. + square))
    similarity = calculate_metric(P, 1. / (1. + absolute))

    return contrast, dissimilarity, homogeneity, similarity


def glcm_props(P):
    # Calculate properties related to correlation

    num_level = P.shape[0]

    I = np.arange(num_level).reshape((num_level, 1, 1, 1))
    J = np.arange(num_level).reshape((1, num_level, 1, 1))

    mean_i = calculate_metric(P, I * P)
    mean_j = calculate_metric(P, J * P)

    diff_i = I - mean_i
    diff_j = J - mean_j

    std_i = np.sqrt(calculate_metric(P, diff_i ** 2))
    std_j = np.sqrt(calculate_metric(P, diff_j ** 2))

    return mean_i, mean_j, diff_i, diff_j, std_i, std_j


def greycoprops_edit(P):
    """Edited version of the scikit-image greycoprops function,
    including additional properties"""

    check_nD(P, 4, 'P')

    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    metrics = defaultdict(np.ndarray)

    # Compute property for each GLCM

    # Compute sum of all squared probabilities
    metrics['ASM'], metrics['energy'] = glcm_product_props(P)

    # Calculate Entropy metrics
    metrics['entropy'] = glcm_log_props(P)

    # Calculate standard weighted metrics
    (metrics['contrast'], metrics['dissimilarity'],
     metrics['homogeneity'], metrics['similarity']) = glcm_difference_props(P)

    # Calculate properties related to correlation
    I = np.arange(num_level).reshape((num_level, 1, 1, 1))
    J = np.arange(num_level).reshape((1, num_level, 1, 1))

    mean_i, mean_j, diff_i, diff_j, std_i, std_j = glcm_props(P)

    metrics['autocorrelation'] = calculate_metric(P, I * J)
    metrics['mean'] = 0.5 * (mean_i + mean_j)
    metrics['covariance'] = calculate_metric(P, diff_i * diff_j)
    metrics['cluster'] = calculate_metric(P, I + J - diff_i - diff_j)

    # Calculate correlation
    correlation = np.zeros((num_dist, num_angle), dtype=np.float64)
    mask_0 = std_i < 1e-15
    mask_1 = ~mask_0

    # handle the standard case
    correlation[mask_1] = (
        metrics['covariance'][mask_1] / (std_i[mask_1] * std_j[mask_1]))

    # handle the special case of standard deviations near zero
    correlation[mask_0] = 1

    metrics['correlation'] = correlation

    return metrics
