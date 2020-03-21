from networkx import Graph

from pyfibre.io.utilities import (
    pop_recursive, remove_contraction,
    deserialize_networkx_graph,
    serialize_networkx_graph
)
from pyfibre.model.tools.metrics import (
    region_shape_metrics, network_metrics, region_texture_metrics)
from pyfibre.model.tools.fibre_assigner import FibreAssigner
from pyfibre.model.tools.fibre_utilities import simplify_network

from .base_graph_segment import BaseGraphSegment
from .fibre import Fibre


class FibreNetwork(BaseGraphSegment):
    """Container for a Networkx Graph and scikit-image region
    representing a connected fibrous region"""

    def __init__(self, *args, fibres=None, red_graph=None, **kwargs):
        super().__init__(*args, **kwargs)

        if fibres is None:
            self.fibres = []
        else:
            self.fibres = [
                Fibre(image=self.image, **element)
                if isinstance(element, dict)
                else element
                for element in fibres
            ]

        if isinstance(red_graph, dict):
            self.red_graph = deserialize_networkx_graph(red_graph)
        elif isinstance(red_graph, Graph):
            self.red_graph = red_graph
        else:
            self.red_graph = None

        self._area_threshold = 200
        self._iterations = 5

    def __getstate__(self):
        """Return the object state in a form that can be
        serialised as a JSON file"""
        state = super().__getstate__()

        if self.red_graph:
            state['red_graph'] = serialize_networkx_graph(state['red_graph'])
            state['red_graph'] = pop_recursive(
                state['red_graph'], remove_contraction)

        state["fibres"] = [
            fibre.__getstate__()
            for fibre in self.fibres]

        return state

    @property
    def fibre_assigner(self):
        return FibreAssigner(
            image=self.image, shape=self.shape)

    def generate_red_graph(self):
        return simplify_network(self.graph)

    def generate_fibres(self):
        return self.fibre_assigner.assign_fibres(self.graph)

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        database = network_metrics(self.graph, self.red_graph, 'SHG')

        shape_metrics = region_shape_metrics(
            self.region, tag='Network')
        database = database.append(
            shape_metrics, ignore_index=False)

        return database
