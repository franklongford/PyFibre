import logging

import networkx as nx
import numpy as np

from pyfibre.utilities import numpy_remove

from pyfibre.model.objects.fibre import Fibre
from .fibre_utilities import (
    distance_matrix, branch_angles,
    get_node_coord_array
)

logger = logging.getLogger(__name__)


class FibreAssigner:

    def __init__(self, image=None, shape=None, angle_thresh=70, min_n=4):

        if image is None and shape is None:
            raise RuntimeError(
                'Cannot instantiate FibreAssigner class'
                ' with both image and shape argument unassigned')

        self.image = image
        self.angle_thresh = angle_thresh
        self.min_n = min_n

        self.graph = None
        self.node_coord = None
        self.edge_count = None
        self.d_coord = None
        self.r2_coord = None

        self._shape = shape

    @property
    def shape(self):
        if self.image is not None:
            return self.image.shape
        return self._shape

    @property
    def theta_thresh(self):
        return np.cos((180 - self.angle_thresh) * np.pi / 180) + 1

    def _get_connected_nodes(self, node):
        """Get nodes connected to input node"""
        return np.array(list(self.graph.adj[node]))

    def _initialise_graph(self, graph):

        self.graph = nx.convert_node_labels_to_integers(graph)
        self.node_coord = get_node_coord_array(self.graph)
        self.edge_count = np.array(
            [self.graph.degree[node] for node in self.graph],
            dtype=int
        )
        self.d_coord, self.r2_coord = distance_matrix(self.node_coord)

    def _create_fibre(self, node):
        """Get nodes connected to input node"""

        new_nodes = self._get_connected_nodes(node)
        edge_list = self.edge_count[new_nodes]

        new_node = new_nodes[np.argsort(edge_list)][-1]
        coord_r = self.graph[node][new_node]['r']

        fibre = Fibre(nodes=[node, new_node],
                      image=self.image,
                      shape=self.shape)

        fibre.graph.nodes[node]['xy'] = self.graph.nodes[node]['xy']
        fibre.graph.nodes[new_node]['xy'] = self.graph.nodes[new_node]['xy']
        fibre.graph.add_edge(
            node, new_node, r=coord_r
        )

        return fibre

    def _grow_fibre(self, fibre):
        """Grow fibre using node"""

        end_node = fibre.node_list[-1]
        new_connect = self._get_connected_nodes(end_node)
        new_connect = numpy_remove(new_connect, fibre.node_list)
        n_edges = new_connect.shape[0]

        if n_edges > 0:
            new_coord_vec = self.d_coord[end_node][new_connect]
            new_coord_r = np.array([
                self.graph[end_node][n]['r'] for n in new_connect
            ])

            assert np.all(new_coord_r > 0), (
                logger.exception(
                    f"{end_node}, {new_connect}, {new_coord_vec},"
                    f" {new_coord_r}, {fibre.node_list}"
                )
            )

            cos_the = branch_angles(fibre.direction, new_coord_vec, new_coord_r)

            try:
                indices = np.argwhere(cos_the + 1 <= self.theta_thresh).flatten()
                straight = (cos_the[indices] + 1).argmin()
                index = indices[straight]

                new_node = new_connect[index]

                fibre.add_node(
                    new_node,
                    xy=self.graph.nodes[new_node]['xy'].copy()
                )
                fibre.add_edge(
                    end_node, new_node,
                    r=self.graph[new_node][end_node]['r'])

            except (ValueError, IndexError):
                fibre.growing = False
        else:
            fibre.growing = False

    def assign_fibres(self, graph):

        self._initialise_graph(graph)
        tracing = np.ones(self.edge_count.shape)
        tot_fibres = []

        for n, node in enumerate(np.argsort(self.edge_count)):
            if tracing[node]:
                fibre = self._create_fibre(node)

                while fibre.growing:
                    self._grow_fibre(fibre)

                if fibre.number_of_nodes >= self.min_n:
                    tot_fibres.append(fibre)
                    for node in fibre.graph:
                        tracing[node] = 0

        return tot_fibres
