"""
ColECM: Collagen ExtraCellular Matrix Simulation
UTILITIES ROUTINE 

Created by: Frank Longford
Created on: 01/11/2015

Last Modified: 12/04/2018
"""
import pickle

import numpy as np

SQRT3 = np.sqrt(3)
SQRT2 = np.sqrt(2)
SQRTPI = np.sqrt(np.pi)


class NoiseError(Exception):

    def __init__(self, noise, thresh):

        self.noise = noise
        self.thresh = thresh
        self.message = "Image too noisy ({} > {})".format(noise, thresh)


def logo():

    logo_text  = "           ___       ___                  " + '\n'
    logo_text += "           |  \\     |   . |              " + '\n'
    logo_text += "           |__/     |__   |__   __  __  " + '\n'
    logo_text += "           |   |  | |   | |  | |   |__| " + '\n'
    logo_text += "           |   \\__| |   | |__/ |   |__  " + '\n'
    logo_text += "                __/                       " + '\n'
    logo_text += "\n    Fibrous Tissue Image Toolkit  v1.5.1a\n"

    return logo_text


def numpy_remove(list1, list2):
    """
    numpy_remove(list1, list2)

    Deletes overlapping elements of list2 from list1
    """

    return np.delete(list1, np.where(np.isin(list1, list2)))


def unit_vector(vector, axis=-1):
    """
    unit_vector(vector, axis=-1)

    Returns unit vector of vector
    """

    vector = np.array(vector)
    magnitude_2 = np.resize(
    np.sum(vector**2, axis=axis), vector.shape)
    u_vector = np.sqrt(vector**2 / magnitude_2) * np.sign(vector)

    return u_vector


def nanmean(array_like, weights=None):

    if weights is None:
        weights = np.ones(array_like.shape)

    indices = ~np.isnan(array_like)

    try:
        average = np.average(
            array_like[indices], weights=weights[indices])
    except ZeroDivisionError:
        average = None

    return average


def ring(image, index, sizes, value):

    index = np.array(index)
    sizes = np.array(sizes)

    for size in sizes:
        indices = np.concatenate((index - size, index + size))

        if indices[0] >= 0:
            image[indices[0],
            max([indices[1], 0]): min([indices[3], image.shape[1]])+1] = value
        if indices[2] < image.shape[0]:
            image[indices[2],
            max([indices[1], 0]): min([indices[3], image.shape[1]])+1] = value
        if indices[1] >= 0:
            image[max([indices[0], 0]): min([indices[2], image.shape[0]])+1,
            indices[1]] = value
        if indices[3] < image.shape[1]:
            image[max([indices[0], 0]): min([indices[2], image.shape[0]])+1,
            indices[3]] = value

    return image


def conv_coord(array):
    return array[:,0], array[:,1]


def clear_border(image, thickness=1):

    for i in range(thickness):
        image[:, 0 + i] = 0
        image[0 + i, :] = 0
        image[:, -(1 + i)] = 0
        image[-(1 + i), :] = 0

    return image


def flatten_list(list_of_lists):

    flat_list = [val for sublist in list_of_lists for val in sublist]

    return flat_list


def matrix_split(matrix, nrows, ncols):
    """Split a matrix into sub-matrices"""

    assert matrix.ndim == 2

    rows = np.array_split(matrix, ncols, axis=0)
    grid = []
    for item in rows:
        grid += np.array_split(item, nrows, axis=-1)

    return grid


def save_pickle(object_, file_name):
    """Saves object as pickled file"""

    with open(file_name, 'wb') as outfile:
        pickle.dump(object_, outfile, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    """Loads pickled object"""

    with open(file_name, 'rb') as infile:
        object_ = pickle.load(infile)

    return object_


def dict_extract(dictionary, keys):

    return {key : dictionary[key] for key in keys}

