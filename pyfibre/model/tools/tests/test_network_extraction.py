import numpy as np
import networkx as nx

from pyfibre.model.tools.network_extraction import (
    NetworkExtraction
)
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestNetworkExtraction(PyFibreTestCase):

    def setUp(self):

        self.N = 20

        self.image = np.zeros((self.N, self.N))
        for i in range(2):
            self.image += 2 * np.eye(self.N, self.N, k=5 - i)
            self.image += np.rot90(np.eye(self.N, self.N, k=5 - i))

        self.network = NetworkExtraction()

    def test___init__(self):

        self.assertIsInstance(self.network.graph, nx.Graph)
        self.assertAlmostEqual(
            0.65797986, self.network.theta_thresh, 6
        )

    def test_theta_thresh(self):

        self.network.angle_thresh = 80
        self.assertAlmostEqual(
            0.82635182, self.network.theta_thresh, 6
        )

    def test__assign_graph(self):

        self.network._assign_graph(nx.Graph())
        self.assertIsInstance(self.network.graph, nx.Graph)
        self.assertEqual(0, self.network.graph.number_of_nodes())

        with self.assertRaises(AssertionError):
            self.network._assign_graph("not a Graph")

    def test__reset_graph(self):

        self.network._reset_graph()
        self.assertIsInstance(self.network.graph, nx.Graph)
        self.assertEqual(0, self.network.graph.number_of_nodes())

    def test__get_connected_nodes(self):

        self.network.graph = nx.grid_2d_graph(2, 3)
        self.assertArrayAlmostEqual(
            np.array([[1, 0], [0, 1]]),
            self.network._get_connected_nodes((0, 0))
        )

    def test__get_nucleation_points(self):

        nuc_node_coord = self.network._get_nucleation_points(self.image)
        self.assertEqual((1, 2), nuc_node_coord.shape)
        self.assertEqual(5, nuc_node_coord[0][0])
        self.assertEqual(10, nuc_node_coord[0][1])
        self.assertIsInstance(nuc_node_coord, np.ndarray)

    def test__initialise_graph(self):
        self.network._initialise_graph(
            self.image, np.array([[5, 10]]))

        self.assertEqual(5, self.network.graph.number_of_nodes())
        self.assertEqual(4, len(self.network.graph.edges))

        self.assertEqual(5.0, self.network.graph[0][1]['r'])
        self.assertEqual(5.0, self.network.graph[0][2]['r'])
        self.assertAlmostEqual(5.65685, self.network.graph[0][3]['r'], 5)
        self.assertAlmostEqual(5.65685, self.network.graph[0][4]['r'], 5)

        for index in range(1, 5):
            node = self.network.graph.nodes[index]
            self.assertTrue(node['growing'])
            self.assertEqual(0, node['nuc'])
            if index == 1:
                self.assertEqual(1, node['xy'][0])
                self.assertEqual(13, node['xy'][1])
                self.assertEqual(-0.8, node['direction'][0])
                self.assertEqual(0.6, node['direction'][1])

            elif index == 2:
                self.assertEqual(2, node['xy'][0])
                self.assertEqual(6, node['xy'][1])
                self.assertEqual(-0.6, node['direction'][0])
                self.assertEqual(-0.8, node['direction'][1])

            elif index == 3:
                self.assertEqual(9, node['xy'][0])
                self.assertEqual(6, node['xy'][1])
                self.assertAlmostEqual(0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(-0.707107, node['direction'][1], 5)

            elif index == 4:
                self.assertEqual(9, node['xy'][0])
                self.assertEqual(14, node['xy'][1])
                self.assertAlmostEqual(0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(0.707106, node['direction'][1], 5)

    def test__get_connections(self):
        self.network._initialise_graph(
            self.image, np.array([[5, 10]]))

        for index in range(1, 5):
            self.assertEqual(1, len(self.network._get_connected_nodes(index)))
            self.assertEqual(0, self.network._get_connected_nodes(index)[0])

    def test_grow_lmp(self):

        self.network._initialise_graph(
            self.image, np.array([[5, 10]]))

        tot_node_coord = [self.network.graph.nodes[node]['xy']
                          for node in self.network.graph]
        tot_node_coord = np.stack(tot_node_coord)

        for index in range(1, 5):
            self.assertTrue(self.network.graph.nodes[index]['growing'])
            self.network.grow_lmp(index, self.image, tot_node_coord)

        self.assertEqual(7, self.network.graph.number_of_nodes())

        for index in range(1, 7):
            node = self.network.graph.nodes[index]
            self.assertEqual(0, node['nuc'])

            if index == 1:
                self.assertEqual(0, node['xy'][0])
                self.assertEqual(15, node['xy'][1])
                self.assertAlmostEqual(-0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(0.707106, node['direction'][1], 5)
                self.assertTrue(node['growing'])

            elif index == 2:
                self.assertEqual(0, node['xy'][0])
                self.assertEqual(5, node['xy'][1])
                self.assertAlmostEqual(-0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(-0.707106, node['direction'][1], 5)
                self.assertTrue(node['growing'])

            elif index == 3:
                self.assertFalse(node['growing'])

            elif index == 4:
                self.assertFalse(node['growing'])

            elif index == 5:
                self.assertEqual(11, node['xy'][0])
                self.assertEqual(4, node['xy'][1])
                self.assertAlmostEqual(0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(-0.707106, node['direction'][1], 5)
                self.assertTrue(node['growing'])

            elif index == 6:
                self.assertEqual(11, node['xy'][0])
                self.assertEqual(16, node['xy'][1])
                self.assertAlmostEqual(0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(0.707106, node['direction'][1], 5)
                self.assertTrue(node['growing'])

    def test_create_network(self):

        self.network.create_network(self.image)
        adjacency = nx.adjacency_matrix(self.network.graph).todense()

        self.assertEqual(8, self.network.graph.number_of_nodes())
        self.assertArrayAlmostEqual(
            np.array([[0, 1, 1, 1, 1, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 1, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0]]),
            adjacency
        )
