import copy

from networkx import Graph

from pyfibre.io.utilities import (
    pop_under_recursive, deserialize_networkx_graph,
    serialize_networkx_graph)
from pyfibre.model.tools.fibre_utilities import get_node_coord_array

from .abc_pyfibre_object import ABCPyFibreObject


class BaseGraph(ABCPyFibreObject):
    """Container for a Networkx Graph representing a connected
     fibrous region on an image"""

    def __init__(self, graph=None):
        if graph is None:
            graph = Graph()
        self.graph = graph

    @property
    def node_list(self):
        return list(self.graph.nodes)

    @property
    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def node_coord(self):
        return get_node_coord_array(self.graph)

    def add_node(self, *args, **kwargs):
        """Add node to Networkx graph attribute"""
        self.graph.add_node(*args, **kwargs)

    def add_edge(self, *args, **kwargs):
        """Add edge to Networkx graph attribute"""
        self.graph.add_edge(*args, **kwargs)

    @classmethod
    def from_json(cls, data):
        graph = data.pop('graph', None)
        if isinstance(graph, dict):
            data['graph'] = deserialize_networkx_graph(graph)
        elif graph is None:
            data['graph'] = Graph()

        return cls(**data)

    def to_json(self):
        """Return the object state in a form that can be
        serialised as a JSON file"""
        state = pop_under_recursive(copy.copy(self.__dict__))
        state['graph'] = serialize_networkx_graph(state['graph'])

        return state

    @classmethod
    def from_array(cls, array, **kwargs):
        """Deserialises numpy array to return an instance
        of the class"""
        raise NotImplementedError(
            f'from_array method not supported for {cls.__class__}')

    def to_array(self, **kwargs):
        """Serialises instance into a numpy array able to be dumped as a
        numpy binary file"""
        raise NotImplementedError(
            f'to_array method not supported for {self.__class__}')
