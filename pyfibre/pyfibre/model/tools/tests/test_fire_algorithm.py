import numpy as np
import networkx as nx

from pyfibre.model.tools.fire_algorithm import FIREAlgorithm
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase


class TestNetworkExtraction(PyFibreTestCase):

    def setUp(self):

        self.N = 20

        self.image = np.zeros((self.N, self.N))
        for i in range(2):
            self.image += 2 * np.eye(self.N, self.N, k=5 - i)
            self.image += np.rot90(np.eye(self.N, self.N, k=5 - i))

        self.fire_algorithm = FIREAlgorithm(
            nuc_thresh=2, lmp_thresh=0.15, angle_thresh=70,
            r_thresh=8, nuc_radius=10)

    def test___init__(self):

        self.assertIsNone(self.fire_algorithm._graph)
        self.assertAlmostEqual(
            0.65797986, self.fire_algorithm.theta_thresh, 6
        )

    def test_theta_thresh(self):

        self.fire_algorithm.angle_thresh = 80
        self.assertAlmostEqual(
            0.82635182, self.fire_algorithm.theta_thresh, 6
        )

    def test__assign_graph(self):

        self.fire_algorithm._assign_graph(nx.Graph())
        self.assertIsInstance(self.fire_algorithm._graph, nx.Graph)
        self.assertEqual(0, self.fire_algorithm._graph.number_of_nodes())

        with self.assertRaises(AssertionError):
            self.fire_algorithm._assign_graph("not a Graph")

    def test__reset_graph(self):

        self.fire_algorithm._reset_graph()
        self.assertIsInstance(self.fire_algorithm._graph, nx.Graph)
        self.assertEqual(0, self.fire_algorithm._graph.number_of_nodes())

    def test__get_connected_nodes(self):

        self.fire_algorithm._graph = nx.grid_2d_graph(2, 3)
        self.assertArrayAlmostEqual(
            np.array([[1, 0], [0, 1]]),
            self.fire_algorithm._get_connected_nodes((0, 0))
        )

    def test__get_nucleation_points(self):

        nuc_node_coord = self.fire_algorithm._get_nucleation_points(self.image)
        self.assertEqual((1, 2), nuc_node_coord.shape)
        self.assertEqual(5, nuc_node_coord[0][0])
        self.assertEqual(10, nuc_node_coord[0][1])
        self.assertIsInstance(nuc_node_coord, np.ndarray)

    def test__initialise_graph(self):
        self.fire_algorithm._graph = nx.Graph()
        self.fire_algorithm._initialise_graph(
            self.image, np.array([[5, 10]]))

        self.assertEqual(5, self.fire_algorithm._graph.number_of_nodes())
        self.assertEqual(4, len(self.fire_algorithm._graph.edges))

        self.assertAlmostEqual(
            5.65685, self.fire_algorithm._graph[0][1]['r'], 5)
        self.assertEqual(5.0, self.fire_algorithm._graph[0][2]['r'])
        self.assertAlmostEqual(
            5.65685, self.fire_algorithm._graph[0][3]['r'], 5)
        self.assertAlmostEqual(
            5.65685, self.fire_algorithm._graph[0][4]['r'], 5)
        self.assertListEqual([1, 2, 3, 4], self.fire_algorithm.grow_list)

        for index in range(1, 5):
            node = self.fire_algorithm._graph.nodes[index]
            self.assertIn(index, self.fire_algorithm.grow_list)
            self.assertEqual(0, node['nuc'])
            if index == 1:
                self.assertEqual(1, node['xy'][0])
                self.assertEqual(14, node['xy'][1])
                self.assertAlmostEqual(-0.707107, node['direction'][0], 5)
                self.assertAlmostEqual(0.707106, node['direction'][1], 5)

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

        self.fire_algorithm._graph = nx.Graph()
        self.fire_algorithm._initialise_graph(
            self.image, np.array([[5, 10]]))

        for index in range(1, 5):
            self.assertEqual(
                1, len(self.fire_algorithm._get_connected_nodes(index)))
            self.assertEqual(
                0, self.fire_algorithm._get_connected_nodes(index)[0])

    def test_grow_lmp(self):

        self.fire_algorithm._graph = nx.Graph()
        self.fire_algorithm._initialise_graph(
            self.image, np.array([[5, 10]]))

        tot_node_coord = [self.fire_algorithm._graph.nodes[node]['xy']
                          for node in self.fire_algorithm._graph]
        tot_node_coord = np.stack(tot_node_coord)

        for index in range(1, 5):
            self.assertIn(index, self.fire_algorithm.grow_list)
            self.fire_algorithm.grow_lmp(index, self.image, tot_node_coord)

        self.assertEqual(7, self.fire_algorithm._graph.number_of_nodes())

        for index in range(1, 7):
            node = self.fire_algorithm._graph.nodes[index]
            self.assertEqual(0, node['nuc'])

            if index == 1:
                self.assertEqual(1, node['xy'][0])
                self.assertEqual(14, node['xy'][1])
                self.assertAlmostEqual(-0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(0.707106, node['direction'][1], 5)
                self.assertNotIn(index, self.fire_algorithm.grow_list)

            elif index == 2:
                self.assertEqual(0, node['xy'][0])
                self.assertEqual(5, node['xy'][1])
                self.assertAlmostEqual(-0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(-0.707106, node['direction'][1], 5)
                self.assertIn(index, self.fire_algorithm.grow_list)

            elif index == 3:
                self.assertNotIn(index, self.fire_algorithm.grow_list)

            elif index == 4:
                self.assertNotIn(index, self.fire_algorithm.grow_list)

            elif index == 5:
                self.assertEqual(11, node['xy'][0])
                self.assertEqual(4, node['xy'][1])
                self.assertAlmostEqual(0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(-0.707106, node['direction'][1], 5)
                self.assertIn(index, self.fire_algorithm.grow_list)

            elif index == 6:
                self.assertEqual(11, node['xy'][0])
                self.assertEqual(16, node['xy'][1])
                self.assertAlmostEqual(0.707106, node['direction'][0], 5)
                self.assertAlmostEqual(0.707106, node['direction'][1], 5)
                self.assertIn(index, self.fire_algorithm.grow_list)

    def test_create_network(self):

        self.fire_algorithm.create_network(self.image)
        adjacency = nx.adjacency_matrix(self.fire_algorithm._graph).todense()

        self.assertEqual(8, self.fire_algorithm._graph.number_of_nodes())
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
