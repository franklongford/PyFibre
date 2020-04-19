import logging
import time
import copy

import networkx as nx
import numpy as np

from skimage.morphology import local_maxima

from pyfibre.utilities import ring, numpy_remove

from .fibre_utilities import (
    branch_angles, reduce_coord, new_branches, transfer_edges,
    check_2D_arrays
)

logger = logging.getLogger(__name__)


class FIREAlgorithm:
    """Class that extracts a complete fibre network from a
    provided image as a single nx.Graph object"""

    def __init__(self, nuc_thresh=2, lmp_thresh=0.15, angle_thresh=70,
                 r_thresh=6, nuc_radius=10):
        """Initialise FibreNetwork object

        Parameters
        ---------
        nuc_thresh : float, optional
            Minimum distance pixel threshold to be classed as
            nucleation point
        lmp_thresh : float, optional
            Minimum distance pixel threshold to be classed as
            lmp point
        angle_thresh : float, optional
            Maximum radian deviation of new lmp from fibre
            trajectory
        r_thresh : float, optional
            Maximum length of edges between nodes
        nuc_radius : float, optional
        """

        self._graph = None
        self.fibres = []
        self.grow_list = []

        self.nuc_thresh = nuc_thresh
        self.lmp_thresh = lmp_thresh
        self.angle_thresh = angle_thresh
        self.r_thresh = r_thresh
        self.nuc_radius = nuc_radius

    @property
    def theta_thresh(self):
        """Conversion of angle_thresh to radians"""
        return 1 + np.cos(
            (180 - self.angle_thresh) * np.pi / 180)

    def _assign_graph(self, graph=None):
        """Assign graph to self.graph"""

        assert isinstance(graph, nx.Graph), (
            f"Argument `graph` must be an object "
            f"of type {nx.Graph}"
        )
        self._graph = graph

    def _reset_graph(self):
        """Reset attribute `graph` to empty nx.Graph object"""
        self._assign_graph(nx.Graph())
        self.grow_list = []
        self.fibres = []

    def _get_connected_nodes(self, node):
        """Get nodes connected to input node"""
        return np.array(list(self._graph.adj[node]))

    def _get_nucleation_points(self, image):
        """Set distance and angle thresholds for fibre iterator"""

        # Get global maxima for smoothed distance matrix
        maxima = local_maxima(
            image, connectivity=self.nuc_radius, allow_borders=True
        )
        nuc_node_coord = reduce_coord(
            np.argwhere(maxima * image >= self.nuc_thresh),
            image[np.where(maxima * image >= self.nuc_thresh)],
            self.r_thresh)

        return nuc_node_coord

    def _initialise_graph(self, image, nuc_node_coord):
        """Initialise graph with nucleation nodes"""

        n_nuc = nuc_node_coord.shape[0]
        self._graph.add_nodes_from(np.arange(n_nuc))
        self.grow_list = list(range(n_nuc))

        n_nodes = n_nuc
        for nuc, nuc_coord in enumerate(nuc_node_coord):

            self._graph.nodes[nuc]['xy'] = nuc_coord
            self._graph.nodes[nuc]['nuc'] = nuc
            self.grow_list.remove(nuc)

            ring_filter = ring(
                np.zeros(image.shape), nuc_coord, [self.r_thresh // 2], 1
            )
            lmp_coord, lmp_vectors, lmp_r = new_branches(
                image, nuc_coord, ring_filter, self.lmp_thresh
            )
            n_lmp = lmp_coord.shape[0]

            self._graph.add_nodes_from(n_nodes + np.arange(n_lmp))
            self._graph.add_edges_from(
                [*zip(nuc * np.ones(n_lmp, dtype=int),
                      n_nodes + np.arange(n_lmp))]
            )

            generator = zip(
                lmp_coord, lmp_vectors, lmp_r, n_nodes + np.arange(n_lmp)
            )

            for xy, vec, r, lmp in generator:
                self._graph.nodes[lmp]['xy'] = xy
                self._graph[nuc][lmp]['r'] = r
                self._graph.nodes[lmp]['nuc'] = nuc
                self.grow_list.append(lmp)
                self._graph.nodes[lmp]['direction'] = -vec / r

            n_nodes += n_lmp

    def grow_lmp(self, index, image, tot_node_coord):
        """
        Grow fibre object along network

        Parameters
        ----------

        index: int
            Index of node to grow on the graph
        image:  array_like, (float); shape=(nx, ny)
            Image to perform FIRE upon
        tot_node_coord: array_like
            Array of full coordinates (x, y) of nodes in graph network
        """

        # Get nodes: end_node (end of fibre), nuc_node (start of fibre)
        # and prior_node (node connected to end)
        end_node = self._graph.nodes[index]
        nuc_node = self._graph.nodes[end_node['nuc']]

        # Get list of connected nodes in fibre
        connected_nodes = self._get_connected_nodes(index)
        prior = connected_nodes[0]
        prior_node = self._graph.nodes[prior]

        # Get edge between end and prior nodes
        edge = self._graph[index][prior]

        ring_filter = ring(
            np.zeros(image.shape), end_node['xy'], np.arange(2, 3), 1
        )

        branch_coord, branch_vector, branch_r = new_branches(
            image, end_node['xy'], ring_filter, self.lmp_thresh
        )

        cos_the = branch_angles(
            end_node['direction'], branch_vector, branch_r
        )
        indices = np.argwhere(abs(cos_the + 1) <= self.theta_thresh)

        if indices.size == 0:
            self.grow_list.remove(index)

            if edge['r'] <= self.r_thresh / 10:
                transfer_edges(self._graph, index, prior)

            return

        branch_coord = branch_coord[indices]
        branch_r = branch_r[indices]

        close_nodes, _ = check_2D_arrays(tot_node_coord, branch_coord, 1)
        close_nodes = numpy_remove(close_nodes, connected_nodes)

        if close_nodes.size != 0:

            new_end = close_nodes.min()
            transfer_edges(self._graph, index, new_end)
            self.grow_list.remove(index)

        else:
            new_index = branch_r.argmax()

            new_end_coord = branch_coord[new_index].flatten()
            new_end_vector = new_end_coord - prior_node['xy']
            new_end_r = np.sqrt((new_end_vector**2).sum())

            new_dir_vector = new_end_coord - nuc_node['xy']
            new_dir_r = np.sqrt((new_dir_vector**2).sum())

            if new_end_r >= self.r_thresh:

                new_end = self._graph.number_of_nodes()

                self._graph.add_node(new_end)
                new_node = self._graph.nodes[new_end]

                self._graph.add_edge(index, new_end)
                new_edge = self._graph[index][new_end]

                new_node['xy'] = new_end_coord
                new_node['nuc'] = end_node['nuc']

                new_edge['r'] = np.sqrt(
                    ((new_end_coord - end_node['xy'])**2).sum())
                new_node['direction'] = (new_dir_vector / new_dir_r)

                self.grow_list.remove(index)
                self.grow_list.append(new_end)

            else:
                end_node['xy'] = new_end_coord
                edge['r'] = new_end_r
                end_node['direction'] = (new_dir_vector / new_dir_r)

    def create_network(self, image):
        """Initialise network from n_nucleation sites"""

        self._reset_graph()

        nuc_node_coord = self._get_nucleation_points(image)

        self._initialise_graph(image, nuc_node_coord)

        n_nuc = nuc_node_coord.shape[0]
        n_node = self._graph.number_of_nodes()

        fibre_grow = []
        fibre_grow[:] = self.grow_list
        n_fibres = len(fibre_grow)

        logger.debug("No. nucleation nodes = {}".format(n_nuc))
        logger.debug("No. nodes created = {}".format(n_node))
        logger.debug("No. fibres to grow = {}".format(n_fibres))

        it = 0
        total_time = 0
        while len(fibre_grow) > 0:
            start = time.time()

            tot_node_coord = [self._graph.nodes[node]['xy']
                              for node in self._graph]
            tot_node_coord = np.stack(tot_node_coord)

            for fibre in fibre_grow:
                self.grow_lmp(
                    fibre, image, tot_node_coord
                )

            n_node = self._graph.number_of_nodes()
            fibre_grow[:] = self.grow_list

            it += 1
            end = time.time()
            total_time += end - start

            logger.debug(
                f"Iteration {it} time = {round(end - start, 3)} s,"
                f" {n_node} nodes  {len(fibre_grow)}/{n_fibres} "
                f"fibres left to grow")

        return copy.copy(self._graph)
