import logging

import networkx as nx
import numpy as np

from pyfibre.utilities import numpy_remove

from pyfibre.model.objects.fibre import Fibre
from .fibre_utilities import (
    distance_matrix, branch_angles,
    get_node_coord_array, get_node_degree_array
)

logger = logging.getLogger(__name__)


class FibreAssigner:
    """Assigns a list of Fibre class instances to a networkx Graph"""

    def __init__(self, angle_thresh=70, min_n=4):

        self.angle_thresh = angle_thresh
        self.min_n = min_n

        self._graph = None
        self.d_coord = None
        self.r_coord = None

    @property
    def theta_thresh(self):
        """Conversion of angle_thresh from degrees into
        cosine(radians)"""
        return np.cos((180 - self.angle_thresh) * np.pi / 180) + 1

    @property
    def node_coord(self):
        """Utility method to generate array of all pixel coordinates
        of networkx graph nodes"""
        if self._graph:
            return get_node_coord_array(self._graph)

    @property
    def edge_count(self):
        """Utility method to generate array containing number of edges
        for each networkx graph node"""
        if self._graph:
            return get_node_degree_array(self._graph)

    def _get_connected_nodes(self, node):
        """Get nodes connected to input node on private graph"""
        return np.array(list(self._graph.adj[node]))

    def _initialise_graph(self, graph):
        """Initialise private networkx graph object"""
        self._graph = nx.convert_node_labels_to_integers(graph)

        # NOTE: Although these could be properties, we generate them
        # only once here once to ensure they are cached. This could be
        # replaced by using cached properties introduced in standard
        # library for Python 3.8
        self.d_coord, r2_coord = distance_matrix(self.node_coord)
        self.r_coord = np.sqrt(r2_coord)

    def _create_fibre(self, node):
        """Create a new Fibre instance beginning from node on
        existing networkx graph"""

        # Find all connected nodes
        new_nodes = self._get_connected_nodes(node)
        edge_list = self.edge_count[new_nodes]

        # Obtain the most connected node as the first connection
        new_node = new_nodes[np.argsort(edge_list)][-1]
        coord_r = self.r_coord[node][new_node]

        fibre = Fibre(nodes=[node, new_node])

        fibre.graph.nodes[node]['xy'] = self.node_coord[node]
        fibre.graph.nodes[new_node]['xy'] = self.node_coord[new_node]
        fibre.graph.add_edge(
            node, new_node, r=coord_r
        )

        return fibre

    def _grow_fibre(self, fibre):
        """Grow fibre using node of networkx graph

        Parameters
        ----------
        fibre: Fibre
            Fibre instance to be grown
        """

        # Obtain node at end of fibre and all connected nodes
        end_node = fibre.node_list[-1]
        new_connect = self._get_connected_nodes(end_node)
        new_connect = numpy_remove(new_connect, fibre.node_list)
        n_edges = new_connect.shape[0]

        # Iterate along connected nodes
        if n_edges > 0:
            # Calculate vectors from end of fibre to new nodes
            new_coord_vec = self.d_coord[end_node][new_connect]
            new_coord_r = np.array([
                self.r_coord[end_node][n] for n in new_connect
            ])

            # Check no connected nodes have the same coordinates
            assert np.all(new_coord_r > 0), (
                logger.exception(
                    f" {end_node}, {new_connect}, "
                    f" {self.node_coord[end_node]}"
                    f" {[self.node_coord[node] for node in new_connect]}"
                    f" {new_coord_vec},"
                    f" {new_coord_r}, {fibre.node_list}"
                )
            )

            # Calculate angles away from current fibre direction
            # to next nodes
            cos_the = branch_angles(
                fibre.direction, new_coord_vec, new_coord_r)

            try:
                # Obtain node that will provide smallest change in
                # direction of Fibre
                indices = np.argwhere(
                    cos_the + 1 <= self.theta_thresh).flatten()
                straight = (cos_the[indices] + 1).argmin()
                index = indices[straight]

                new_node = new_connect[index]

                # Add node to end of growing Fibre
                fibre.add_node(
                    new_node,
                    xy=self.node_coord[new_node]
                )
                fibre.add_edge(
                    end_node, new_node,
                    r=self.r_coord[new_node][end_node])

            except (ValueError, IndexError):
                fibre.growing = False
        else:
            fibre.growing = False

    def assign_fibres(self, graph):
        """Returns a list of Fibre instances, extracted from a networkx graph

        Parameters
        ----------
        graph: Networkx.Graph
            Graph to extract fibres from

        Returns
        -------
        tot_fibres: list of Fibre
            List of fibre objects extracted from graph

        """
        self._initialise_graph(graph)
        tracing = np.ones(self.edge_count.shape)
        tot_fibres = []

        # Iterate through all nodes at network ends
        for n, node in enumerate(np.argsort(self.edge_count)):
            if tracing[node]:

                # Create new Fibre instance and grow by tracing along
                # existing networkx graph
                fibre = self._create_fibre(node)

                while fibre.growing:
                    self._grow_fibre(fibre)

                # If Fibre is long enough, add it to the list and
                # prevent nodes that are contained within it from
                # ending up in any new Fibre
                if fibre.number_of_nodes >= self.min_n:
                    tot_fibres.append(fibre)
                    for node in fibre.graph:
                        tracing[node] = 0

        return tot_fibres
