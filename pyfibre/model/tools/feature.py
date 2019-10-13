"""
PyFibre
Image Tools Library

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 18/02/2019
"""

import numpy as np

from skimage.feature import greycoprops


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