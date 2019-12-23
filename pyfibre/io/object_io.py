import numpy as np
from networkx.readwrite import node_link_graph

from pyfibre.model.objects.base_graph_segment import BaseGraphSegment
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.utilities import (
    save_pickle, load_pickle, save_json, load_json)


def get_networkx_graph(data):
    """Transform JSON serialised data into a
    networkx Graph object"""

    for coord in data['nodes']:
        coord['xy'] = np.asarray(coord['xy'])
        if 'direction' in coord:
            coord['direction'] = np.asarray(coord['direction'])

    return node_link_graph(data)


def save_base_graph_segment(graph_segment, file_name, file_type=None):

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    data = graph_segment.__getstate__()

    save_json(data, file_name)


def load_base_graph_segment(file_name, file_type=None, klass=BaseGraphSegment):
    """Loads pickled image objects"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    data = load_json(file_name)

    data['graph'] = get_networkx_graph(data['graph'])
    data['image'] = np.asarray(data['image'])

    return klass(**data)


def save_base_graph_segments(graph_segments, file_name, file_type=None):

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])
    else:
        file_type = 'graph_segment'

    data = {
        file_type : [
            graph_segment.__getstate__()
            for graph_segment in graph_segments]
    }

    save_json(data, file_name)


def load_base_graph_segments(file_name, file_type=None, klass=BaseGraphSegment):

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])
    else:
        file_type = 'graph_segment'

    data = load_json(file_name)

    for graph_segment in data[file_type]:
        graph_segment['graph'] = get_networkx_graph(
            graph_segment['graph'])
        graph_segment['image'] = np.asarray(
            graph_segment['image'])

    graph_segments = [
        klass(**kwargs) for kwargs in data[file_type]
    ]

    return graph_segments


def save_fibre_networks(fibre_networks, file_name):
    """Save a list of FibreNetwork instances"""
    save_base_graph_segments(fibre_networks, file_name, 'fibre_networks')


def load_fibre_networks(file_name):
    """Load a list of FibreNetwork instances"""
    return load_base_graph_segments(
        file_name, 'fibre_networks', FibreNetwork)


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

    try:
        objects = load_pickle(f"{file_name}.pkl")
        return objects
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}.pkl"
        ) from e
