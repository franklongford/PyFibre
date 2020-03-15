import copy

from networkx import Graph

from pyfibre.io.utilities import (
    pop_under_recursive, deserialize_networkx_graph,
    serialize_networkx_graph)
from pyfibre.model.tools.fibre_utilities import get_node_coord_array


class BaseGraph:
    """Container for a Networkx Graph representing a connected
     fibrous region on an image"""

    def __init__(self, graph=None, image=None, shape=None):

        if image is None and shape is None:
            raise AttributeError(
                'Cannot instantiate BaseGraph class: '
                'either image or shape argument must be declared')

        if isinstance(graph, dict):
            graph = deserialize_networkx_graph(graph)
        elif graph is None:
            graph = Graph()

        self.graph = graph
        self.image = image

        self._shape = shape

    def __getstate__(self):
        """Return the object state in a form that can be
        serialised as a JSON file"""
        state = pop_under_recursive(copy.copy(self.__dict__))

        state.pop('image', None)
        state['shape'] = self.shape
        state['graph'] = serialize_networkx_graph(state['graph'])

        return state

    @property
    def node_list(self):
        return list(self.graph.nodes)

    @property
    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def node_coord(self):
        return get_node_coord_array(self.graph)

    @property
    def shape(self):
        if self.image is not None:
            return self.image.shape
        return self._shape

    def add_node(self, *args, **kwargs):
        """Add node to Networkx graph attribute"""
        self.graph.add_node(*args, **kwargs)

    def add_edge(self, *args, **kwargs):
        """Add edge to Networkx graph attribute"""
        self.graph.add_edge(*args, **kwargs)

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""
        raise NotImplementedError()
