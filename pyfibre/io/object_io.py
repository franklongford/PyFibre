from pyfibre.io.utils import get_networkx_graph
from pyfibre.model.objects.base_graph_segment import BaseGraphSegment
from pyfibre.model.objects.cell import Cell
from pyfibre.model.objects.fibre import Fibre
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.utilities import (
    save_pickle, load_pickle, save_json, load_json)

from .segment_io import save_segments, load_segments


def save_base_graph_segment(graph_segment, file_name, file_type=None):

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    data = graph_segment.__getstate__()

    save_json(data, file_name)


def load_base_graph_segment(
        file_name, file_type=None, klass=BaseGraphSegment, image=None):
    """Loads pickled image objects"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    data = load_json(file_name)
    data['graph'] = get_networkx_graph(data['graph'])
    data['image'] = image

    return klass(**data)


def save_base_graph_segments(graph_segments, file_name, file_type=None):

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])
    else:
        file_type = 'graph_segment'

    data = {file_type: []}

    for graph_segment in graph_segments:
        if isinstance(graph_segment, BaseGraphSegment):
            data["file_type"].append(
                graph_segment.__getstate__())
        else:
            data["file_type"] += [
                element.__getstate__()
                for element in graph_segment
            ]

    save_json(data, file_name)


def load_base_graph_segments(
        file_name, file_type=None, klass=BaseGraphSegment, image=None):

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])
    else:
        file_type = 'graph_segment'

    data = load_json(file_name)

    for graph_segment in data[file_type]:
        if isinstance(graph_segment, dict):
            graph_segment['graph'] = get_networkx_graph(
                graph_segment['graph'])
            graph_segment['image'] = image
        else:
            for element in graph_segment:
                graph_segment['graph'] = get_networkx_graph(
                    graph_segment['graph'])
                graph_segment['image'] = image


    graph_segments = [
        klass(**kwargs) for kwargs in data[file_type]
    ]

    return graph_segments


def save_fibres(fibres, file_name):
    """Save a nested list of Fibre instances"""
    save_base_graph_segments(fibres, file_name, 'fibres')


def load_fibres(file_name, image=None):
    """Load a nested list of FibreNetwork instances"""
    return load_base_graph_segments(
        file_name, 'fibres', Fibre, image=image)


def save_cells(cells, file_name, shape=None):
    """Save a list of Cell instances"""
    segments = [cell.segment for cell in cells]
    if shape is None:
        shape = cells[0].segment.image.shape
    save_segments(segments, file_name, shape, 'cells')


def load_cells(file_name, image=None):
    """Load a list of Cell instances"""
    segments = load_segments(
        file_name, 'cells', image=image)
    return [Cell(segment=segment)
            for segment in segments]


def save_fibre_networks(fibre_networks, file_name):
    """Save a list of FibreNetwork instances"""
    save_base_graph_segments(fibre_networks, file_name, 'fibre_networks')


def load_fibre_networks(file_name, image=None):
    """Load a list of FibreNetwork instances"""
    return load_base_graph_segments(
        file_name, 'fibre_networks', FibreNetwork, image=image)


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
