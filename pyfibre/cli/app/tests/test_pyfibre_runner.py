from unittest import TestCase

from pyfibre.shg_pl_trans.tests.probe_classes import ProbeSHGPLTransImage
from pyfibre.tests.probe_classes.objects import (
    ProbeFibreNetwork, ProbeFibreSegment, ProbeCellSegment)

from pyfibre.cli.app.pyfibre_runner import PyFibreRunner


LOAD_NETWORK_PATH = "networkx.read_gpickle"
SAVE_NETWORK_PATH = "networkx.write_gpickle"

LOAD_JSON_PATH = "json.load"
SAVE_JSON_PATH = "json.dump"

LOAD_REGION_PATH = "numpy.load"
SAVE_REGION_PATH = "numpy.save"


def mock_load(*args, klass=None, **kwargs):
    print('mock_load called')
    return klass()


class TestPyFibreRunner(TestCase):

    def setUp(self):

        self.multi_image = ProbeSHGPLTransImage()
        self.runner = PyFibreRunner()
        self.fibre_networks = [ProbeFibreNetwork()]

        self.fibre_segments = [ProbeFibreSegment()]
        self.cell_segments = [ProbeCellSegment()]

    def test_defaults(self):
        self.assertEqual((5, 35), self.runner.p_denoise)
        self.assertDictEqual(
            {'nuc_thresh': 2,
             'nuc_radius': 11,
             'lmp_thresh': 0.15,
             'angle_thresh': 70,
             'r_thresh': 7},
            self.runner.fire_parameters)

    def test_run_analysis(self):
        pass
