import numpy as np
import networkx as nx
from dataclasses import dataclass

from pyfibre.tools.fibre_utilities import branch_angles, get_node_coord_array


@dataclass(kw_only=True)
class Fibre:
    """Container for a Networkx Graph and scikit-image region
    representing a single, un-branched fibre"""

    graph: nx.Graph

    growing: bool = True

    @property
    def node_list(self):
        """Helper routine to return a list of node labels in
        the networkx graph"""
        return list(self.graph.nodes)

    @property
    def node_coord(self):
        """Helper routine to return a numpy array of pixel coordinates
        of each node in the networkx graph."""
        return get_node_coord_array(self.graph)

    @property
    def _d_coord(self):
        try:
            return self.node_coord[-1] - self.node_coord[0]
        except Exception:
            return np.array([0, 0])

    @property
    def euclid_l(self):
        """Euclidean distance between each end of the fibre"""
        return np.sqrt(np.sum(self._d_coord**2))

    @property
    def fibre_l(self):
        """Perimeter distance along entire fibre"""
        try:
            fibre_l = [self.graph[edge[0]][edge[1]]["r"] for edge in self.graph.edges]
            return sum(fibre_l)
        except Exception:
            return 0

    @property
    def direction(self):
        """Vector representing the direction of fibre"""
        if self.euclid_l > 0:
            return -self._d_coord / self.euclid_l
        return np.array([0, 0])

    @property
    def angle(self):
        """Angle relating to fibre direction"""
        cos_the = branch_angles(self.direction, np.array([[0, 1]]), np.ones(1))[0]
        return 180 / np.pi * np.arccos(cos_the)

    @property
    def waviness(self):
        """Normalised metric representing the fibre waviness;
        euclidean distance divided by fibre perimeter"""
        if self.fibre_l > 0:
            return self.euclid_l / self.fibre_l
        return np.nan


def fiber_from_node_list(nodes: list[int], edges: list[tuple[int, int]] | None = None):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    if edges is not None:
        graph.add_edges_from(edges)
    return Fibre(graph=graph)
