from functools import partial

import numpy as np

from pyfibre.model.objects.abc_pyfibre_object import ABCPyFibreObject
from pyfibre.model.objects.segments import CellSegment, FibreSegment
from pyfibre.model.objects.fibre import Fibre
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.io.utilities import (
    save_json, load_json, save_numpy, load_numpy)


SUPPORTED_MODES = ['json', 'array']


def create_file_name(file_name, file_type):
    if file_type is not None:
        return '_'.join([file_name, file_type])
    return file_name


def save_pyfibre_object(pyfibre_object, file_name, mode,
                        file_type=None, **kwargs):
    """Save an ABCPyFibreObject subclass"""

    if mode not in SUPPORTED_MODES:
        raise AttributeError(f'Save mode {mode} not supported')

    file_name = create_file_name(file_name, file_type)

    if mode == 'json':
        data = pyfibre_object.to_json()
        save_json(data, file_name)
    else:
        array = pyfibre_object.to_array(**kwargs)
        save_numpy(file_name, array)


def load_pyfibre_object(file_name, klass, mode,
                        file_type=None, **kwargs):
    """Load an ABCPyFibreObject subclass"""

    if not issubclass(klass, ABCPyFibreObject):
        raise TypeError(
            'klass argument must be of type ABCPyFibreObject')

    if mode not in SUPPORTED_MODES:
        raise AttributeError(f'Save mode {mode} not supported')

    file_name = create_file_name(file_name, file_type)

    if mode == 'json':
        data = load_json(file_name)
        data.update(kwargs)
        return klass.from_json(data)
    else:
        array = load_numpy(file_name)
        return klass.from_array(array, **kwargs)


def save_pyfibre_objects(pyfibre_objects, file_name, mode,
                         file_type=None, **kwargs):
    """Save a list of ABCPyFibreObject subclass"""

    if mode not in SUPPORTED_MODES:
        raise AttributeError(f'Save mode {mode} not supported')

    file_name = create_file_name(file_name, file_type)

    if file_type is None:
        file_type = 'pyfibre_objects'

    if mode == 'json':
        data = {
            file_type: [
                pyfibre_object.to_json()
                for pyfibre_object in pyfibre_objects
            ]
        }
        save_json(data, file_name)
    else:
        try:
            shape = kwargs['shape']
        except KeyError:
            shape = pyfibre_objects[0].region.image.shape

        stack = np.zeros(
            (len(pyfibre_objects),) + shape, dtype=np.int)

        for index, pyfibre_object in enumerate(pyfibre_objects):
            array = pyfibre_object.to_array(shape=shape)
            stack[index] += array
        stack = np.where(stack, 1, 0)

        save_numpy(file_name, stack)


def load_pyfibre_objects(file_name, klass, mode,
                         file_type=None, **kwargs):
    """Load a list of ABCPyFibreObject subclass"""
    if not issubclass(klass, ABCPyFibreObject):
        raise TypeError(
            'klass argument must be of type ABCPyFibreObject')

    if mode not in SUPPORTED_MODES:
        raise AttributeError(f'Save mode {mode} not supported')

    file_name = create_file_name(file_name, file_type)

    if file_type is None:
        file_type = 'pyfibre_objects'

    pyfibre_objects = []

    if mode == 'json':
        data = load_json(file_name)
        for data in data[file_type]:
            data.update(kwargs)
            pyfibre_objects.append(klass.from_json(data))
    else:
        stack = load_numpy(file_name)
        for array in stack:
            pyfibre_objects.append(
                klass.from_array(array, **kwargs))

    return pyfibre_objects


save_fibres = partial(
    save_pyfibre_objects, mode='json', file_type='fibres')

load_fibres = partial(
    load_pyfibre_objects, mode='json', klass=Fibre, file_type='fibres')

save_fibre_networks = partial(
    save_pyfibre_objects, mode='json', file_type='fibre_networks')

load_fibre_networks = partial(
    load_pyfibre_objects, mode='json', klass=FibreNetwork,
    file_type='fibre_networks')

save_cell_segments = partial(
    save_pyfibre_objects, mode='array', file_type='cell_segments'
)

load_cell_segments = partial(
    load_pyfibre_objects, mode='array', klass=CellSegment,
    file_type='cell_segments'
)

save_fibre_segments = partial(
    save_pyfibre_objects, mode='array', file_type='fibre_segments'
)

load_fibre_segments = partial(
    load_pyfibre_objects, mode='array', klass=FibreSegment,
    file_type='fibre_segments'
)
