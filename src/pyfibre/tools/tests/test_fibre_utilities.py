import numpy as np
from unittest import TestCase

from pyfibre.tools.fibre_utilities import (
    check_2D_arrays,
    distance_matrix,
    branch_angles,
    remove_redundant_nodes,
    transfer_edges,
    get_edge_list,
    simplify_network,
    reduce_coord,
)
from pyfibre.testing.example_objects import generate_image, generate_probe_graph


class TestFibreUtilities(TestCase):
    def setUp(self):
        self.image, _, _, _ = generate_image()
        self.network = generate_probe_graph()
        self.pos_2D = np.array([[1, 3], [4, 2], [1, 5]])
        self.answer_d_2D = np.array(
            [
                [[0, 0], [3, -1], [0, 2]],
                [[-3, 1], [0, 0], [-3, 3]],
                [[0, -2], [3, -3], [0, 0]],
            ]
        )
        self.answer_r2_2D = np.array([[0, 10, 4], [10, 0, 18], [4, 18, 0]])
        self.answer_cos_the = np.array([0, 0.9486833])

    def test_check_2D_arrays(self):
        indices = check_2D_arrays(self.pos_2D, self.pos_2D + 1.5, 2)
        self.assertEqual(indices[0], 2)
        self.assertEqual(indices[1], 0)

    def test_distance_matrix(self):
        d_2D, r2_2D = distance_matrix(self.pos_2D)

        np.testing.assert_almost_equal(d_2D, self.answer_d_2D)
        np.testing.assert_almost_equal(r2_2D, self.answer_r2_2D)

    def test_get_edge_list(self):
        edge_list = get_edge_list(self.network)
        np.testing.assert_almost_equal(edge_list, np.array([[2, 3], [3, 4], [5, 4]]))

    def test_branch_angles(self):
        direction = np.array([1, 0])
        vectors = self.answer_d_2D[([2, 0], [0, 1])]
        r = np.sqrt(self.answer_r2_2D[([2, 0], [0, 1])])

        cos_the = branch_angles(direction, vectors, r)
        np.testing.assert_almost_equal(self.answer_cos_the, cos_the)

    def test_transfer_edges(self):
        transfer_edges(self.network, 3, 2)

        self.assertEqual(1, self.network.degree[2])
        self.assertEqual(0, self.network.degree[3])
        self.assertEqual(2, self.network.degree[4])
        self.assertEqual(1, self.network.degree[5])

    def test_remove_redundant_nodes(self):
        network = remove_redundant_nodes(self.network, 0.5)
        self.assertListEqual([0, 1, 2, 3], list(network.nodes))
        self.assertEqual(1, network.degree[0])
        self.assertEqual(2, network.degree[1])
        self.assertEqual(2, network.degree[2])
        self.assertEqual(1, network.degree[3])

        network = remove_redundant_nodes(self.network, 1.2)
        self.assertListEqual([0, 1, 2], list(network.nodes))
        self.assertEqual(1, network.degree[0])
        self.assertEqual(2, network.degree[1])
        self.assertEqual(1, network.degree[2])

        network = remove_redundant_nodes(self.network)
        self.assertListEqual([0], list(network.nodes))
        self.assertEqual(0, network.degree[0])

    def test_simplify_network(self):
        network = simplify_network(self.network)
        self.assertListEqual([0, 1], list(network.nodes))

    def test_reduce_coord(self):
        coord = np.argwhere(self.image > 2)
        weights = np.arange(len(coord))

        reduced_coord = reduce_coord(coord)
        np.testing.assert_almost_equal(np.array([[2, 7], [8, 3]]), reduced_coord)

        reduced_coord = reduce_coord(coord, weights)
        np.testing.assert_almost_equal(np.array([[2, 4], [8, 1]]), reduced_coord)

        # Edge case with no minimum threshold
        reduced_coord = reduce_coord(coord, weights, thresh=0)
        np.testing.assert_almost_equal(coord, reduced_coord)

        # Edge case with infinite threshold, should always
        # return at least one coordinate
        reduced_coord = reduce_coord(coord, weights, thresh=500)
        np.testing.assert_almost_equal(np.array([[2, 4]]), reduced_coord)
