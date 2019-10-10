from unittest import TestCase

import numpy as np

from pyfibre.model.tools.fibre_assignment import (
    Fibre, FibreAssignment
)
from pyfibre.tests.probe_classes import generate_probe_network


class TestFibreAssignment(TestCase):

    def setUp(self):

        self.fibre_assignment = FibreAssignment()
        self.graph = generate_probe_network()

    def test___init__(self):

        self.assertIsNone(self.fibre_assignment.graph)
        self.assertAlmostEqual(
            0.65797986, self.fibre_assignment.theta_thresh, 6
        )

    def test__initialise_graph(self):

        self.fibre_assignment._initialise_graph(self.graph)
        self.assertEqual([0, 1, 2, 3], list(self.fibre_assignment.graph.nodes))
        self.assertTrue(np.allclose(
            np.array([[0, 0], [1, 1], [2, 2], [3, 3.2]]),
            self.fibre_assignment.node_coord
        ))
        self.assertTrue(np.allclose(
            np.array([1, 2, 2, 1]),
            self.fibre_assignment.edge_count
        ))

        self.assertTrue(np.allclose(
            np.array([[0, 1, 2, 3],
                      [-1, 0, 1, 2],
                      [-2, -1, 0, 1],
                      [-3, -2, -1, 0]]),
            self.fibre_assignment.d_coord[...,0]
        ))

        self.assertTrue(np.allclose(
            np.array([[0, 1, 2, 3.2],
                      [-1, 0, 1, 2.2],
                      [-2, -1, 0, 1.2],
                      [-3.2, -2.2, -1.2, 0]]),
            self.fibre_assignment.d_coord[...,1]
        ))

        self.assertTrue(np.allclose(
            np.array([[0, 2, 8, 19.24],
                      [2, 0, 2, 8.84],
                      [8, 2, 0, 2.44],
                      [19.24, 8.84, 2.44, 0]]),
            self.fibre_assignment.r2_coord
        ))

    def test_theta_thresh(self):

        self.fibre_assignment.angle_thresh = 80
        self.assertAlmostEqual(
            0.82635182, self.fibre_assignment.theta_thresh, 6
        )

    def test__create_fibre(self):

        self.fibre_assignment._initialise_graph(self.graph)
        fibre = self.fibre_assignment._create_fibre(0)

        self.assertIsInstance(fibre, Fibre)
        self.assertEqual([0, 1], fibre.node_list)
        self.assertTrue(np.allclose(
            np.array([-0.70710678, -0.70710678]),
            fibre.direction
        ))
        self.assertTrue(fibre.growing)
        self.assertEqual(np.sqrt(2), fibre.fibre_l)
        self.assertEqual(np.sqrt(2), fibre.euclid_l)

    def test__grow_fibre(self):

        self.fibre_assignment._initialise_graph(self.graph)
        fibre = self.fibre_assignment._create_fibre(0)

        self.fibre_assignment._grow_fibre(fibre, 2)
        self.assertEqual([0, 1, 2], fibre.node_list)
        self.assertTrue(np.allclose(
            np.array([-0.70710678, -0.70710678]),
            fibre.direction
        ))
        self.assertTrue(fibre.growing)
        self.assertAlmostEqual(2.82842712, fibre.fibre_l, 5)
        self.assertAlmostEqual(2.82842712, fibre.euclid_l, 5)

        self.fibre_assignment._grow_fibre(fibre, 3)
        self.assertEqual([0, 1, 2, 3], fibre.node_list)
        self.assertTrue(np.allclose(
            np.array([-0.68394113, -0.7295372]),
            fibre.direction
        ))
        self.assertTrue(fibre.growing)
        self.assertAlmostEqual(4.24264068, fibre.fibre_l, 5)
        self.assertAlmostEqual(4.38634244, fibre.euclid_l, 5)

    def test_assign_fibres(self):

        tot_fibres = self.fibre_assignment.assign_fibres(self.graph)

        self.assertEqual(1, len(tot_fibres))

        fibre = tot_fibres[0]
        self.assertEqual([0, 1, 2, 3], fibre.node_list)
        self.assertTrue(np.allclose(
            np.array([-0.68394113, -0.7295372]),
            fibre.direction
        ))
        self.assertFalse(fibre.growing)
        self.assertAlmostEqual(4.24264068, fibre.fibre_l, 5)
        self.assertAlmostEqual(4.38634244, fibre.euclid_l, 5)
