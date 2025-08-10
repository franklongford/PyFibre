import networkx as nx

from dataclasses import dataclass
from pyfibre.fibers.fibre_assigner import FibreAssigner
from pyfibre.fibers.fibre_utilities import simplify_network


@dataclass(frozen=True, kw_only=True)
class FibreNetwork:
    """Container for a Networkx Graph
    representing a connected fibrous region"""

    graph: nx.Graph

    @property
    def node_list(self):
        """Helper routine to return a list of node labels in
        the networkx graph"""
        return list(self.graph.nodes)

    def generate_red_graph(self):
        return simplify_network(self.graph)

    def generate_fibres(self):
        return FibreAssigner().assign_fibres(self.graph)
