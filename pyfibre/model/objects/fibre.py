import pandas as pd
import networkx as nx
import numpy as np

from pyfibre.model.tools.analysis import segment_analysis
from pyfibre.model.tools.fibre_utilities import branch_angles

from .base_graph_segment import BaseGraphSegment


class Fibre(BaseGraphSegment):
    """Container for a Networkx Graph and scikit-image region
    representing a single, un-branched fibre"""

    def __init__(self, *args, graph=None, nodes=None, edges=None, growing=True, **kwargs):

        if not isinstance(graph, nx.Graph):
            graph = nx.Graph()
            if nodes is not None:
                graph.add_nodes_from(nodes)
            if edges is not None:
                graph.add_edges_from(edges)

        super().__init__(*args, graph=graph, **kwargs)

        self.growing = growing

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
            fibre_l = [
                self.graph[edge[0]][edge[1]]['r']
                for edge in self.graph.edges
            ]
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
        cos_the = branch_angles(
            self.direction, np.array([[0, 1]]), np.ones(1))[0]
        return 180 / np.pi * np.arccos(cos_the)

    @property
    def waviness(self):
        """Normalised metric representing the fibre waviness;
        euclidean distance divided by fibre perimeter"""
        if self.fibre_l > 0:
            return self.euclid_l / self.fibre_l
        return np.nan

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        database = pd.Series()

        database['Fibre Waviness'] = self.waviness
        database['Fibre Length'] = self.fibre_l
        database['Fibre Angle'] = self.angle

        if image is not None:
            segment_metrics = segment_analysis(
                self.segment, image=image, tag='Fibre')

        else:
            segment_metrics = segment_analysis(
                self.segment, tag='Fibre')

        database = database.append(segment_metrics, ignore_index=False)

        return database