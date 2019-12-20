import numpy as np
from networkx.readwrite import node_link_graph

from pyfibre.model.objects.base_graph_segment import BaseGraphSegment
from pyfibre.utilities import (
    save_pickle, load_pickle, save_json, load_json)


def save_base_graph_segment(object, file_name, file_type=None):

    data = object.__getstate__()

    try:
        save_json(data, f"{file_name}.json")
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}.json"
        ) from e


def load_base_graph_segment(file_name, file_type=None):
    """Loads pickled image objects"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        data = load_json(f"{file_name}.json")

    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}.json"
        ) from e

    for coord in data['graph']['nodes']:
        coord['xy'] = np.array(coord['xy'])

    data['graph'] = node_link_graph(data['graph'])

    return BaseGraphSegment(**data)


def save_objects(objects, file_name, file_type=None):
    """Loads pickled image objects"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        save_pickle(objects, f"{file_name}.pkl")
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}.pkl"
        ) from e


def load_objects(file_name, file_type=None):
    """Loads pickled image objects"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    print(file_name)
    try:
        objects = load_pickle(f"{file_name}.pkl")
        return objects
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}.pkl"
        ) from e
