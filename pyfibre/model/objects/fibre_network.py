from networkx import Graph

from pyfibre.io.utilities import (
    pop_recursive, remove_contraction,
    deserialize_networkx_graph,
    serialize_networkx_graph
)
from pyfibre.model.tools.metrics import (
    network_metrics, fibre_metrics, FIBRE_METRICS)
from pyfibre.model.tools.fibre_assigner import FibreAssigner
from pyfibre.model.tools.fibre_utilities import simplify_network

from .base_graph_segment import BaseGraph
from .fibre import Fibre


class FibreNetwork(BaseGraph):
    """Container for a Networkx Graph
    representing a connected fibrous region"""

    def __init__(self, *args, fibres=None, red_graph=None, **kwargs):
        super(FibreNetwork, self).__init__(*args, **kwargs)

        if fibres is None:
            fibres = []
        self.fibres = fibres

        self.red_graph = red_graph

    @property
    def fibre_assigner(self):
        return FibreAssigner()

    def generate_red_graph(self):
        return simplify_network(self.graph)

    def generate_fibres(self):
        return self.fibre_assigner.assign_fibres(self.graph)

    @classmethod
    def from_json(cls, data):

        fibres = data.pop('fibres', None)
        if fibres is None:
            data['fibres'] = []
        else:
            data['fibres'] = [
                Fibre.from_json(kwargs)
                if isinstance(kwargs, dict)
                else kwargs
                for kwargs in fibres
            ]

        for attr in ['graph', 'red_graph']:
            graph = data.pop(attr, None)
            if isinstance(graph, dict):
                data[attr] = deserialize_networkx_graph(graph)
            elif graph is None:
                data[attr] = Graph()

        return cls(**data)

    def to_json(self):
        """Return the object state in a form that can be
        serialised as a JSON file"""
        state = super(FibreNetwork, self).to_json()

        if self.red_graph:
            state['red_graph'] = serialize_networkx_graph(state['red_graph'])
            state['red_graph'] = pop_recursive(
                state['red_graph'], remove_contraction)

        state["fibres"] = [
            fibre.to_json() for fibre in self.fibres]

        return state

    def generate_database(self):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        database = network_metrics(
            self.graph, self.red_graph, len(self.fibres), 'Fibre')

        metrics = fibre_metrics(self.fibres)
        mean_metrics = metrics.mean()

        for metric in FIBRE_METRICS:
            database[f'Mean Fibre {metric}'] = (
                mean_metrics[f'Fibre {metric}'])

        return database
