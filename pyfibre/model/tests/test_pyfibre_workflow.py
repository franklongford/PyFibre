from unittest import TestCase

from pyfibre.model.pyfibre_workflow import PyFibreWorkflow


class TestPyFibreWorkflow(TestCase):

    def setUp(self):
        self.workflow = PyFibreWorkflow()

    def test_defaults(self):
        self.assertEqual((5, 35), self.workflow.p_denoise)
        self.assertDictEqual(
            {'nuc_thresh': 2,
             'nuc_radius': 11,
             'lmp_thresh': 0.15,
             'angle_thresh': 70,
             'r_thresh': 7},
            self.workflow.fire_parameters)
