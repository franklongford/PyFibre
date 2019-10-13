import networkx as nx
import numpy as np


class Fibre(nx.Graph):
    """Networkx Graph object representing a single, un-branched fibre"""

    def __init__(self, graph=None, nodes=None, edges=None, growing=True):

        if not isinstance(graph, nx.Graph):
            graph = nx.Graph()
            if nodes is not None:
                graph.add_nodes_from(nodes)
            if edges is not None:
                graph.add_edges_from(edges)

        super(Fibre, self).__init__(graph)
        self.growing = growing

    @property
    def node_list(self):
        return list(self.nodes)

    @property
    def _d_coord(self):
        try:
            start = self.node_list[0]
            end = self.node_list[-1]
            return self.nodes[end]['xy'] - self.nodes[start]['xy']
        except Exception:
            return np.array([0, 0])

    @property
    def euclid_l(self):
        return np.sqrt(np.sum(self._d_coord**2))

    @property
    def fibre_l(self):
        try:
            fibre_l = [
                self[edge[0]][edge[1]]['r']
                for edge in self.edges
            ]
            return sum(fibre_l)
        except Exception:
            return 0

    @property
    def direction(self):
        if self.euclid_l > 0:
            return -self._d_coord / self.euclid_l
        return np.array([0, 0])

    @property
    def waviness(self):
        if self.fibre_l > 0:
            return self.euclid_l / self.fibre_l
        return np.nan
