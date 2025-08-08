import networkx as nx

from dataclasses import dataclass
from pyfibre.fibers.fibre_assigner import FibreAssigner
from pyfibre.tools.fibre_utilities import simplify_network


@dataclass(frozen=True, kw_only=True)
class FibreNetwork:
    """Container for a Networkx Graph
    representing a connected fibrous region"""

    graph: nx.Graph

    def generate_red_graph(self):
        return simplify_network(self.graph)

    def generate_fibres(self):
        return FibreAssigner().assign_fibres(self.graph)
