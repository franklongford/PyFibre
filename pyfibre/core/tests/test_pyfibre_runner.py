from unittest import TestCase

from pyfibre.model.objects.segments import (
    FibreSegment, CellSegment
)
from pyfibre.tests.probe_classes.objects import (
    ProbeFibreNetwork, generate_probe_segment)

from pyfibre.core.pyfibre_runner import PyFibreRunner


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

        self.runner = PyFibreRunner()
        self.fibre_networks = [ProbeFibreNetwork()]

        self.fibre_segments = [
            generate_probe_segment(FibreSegment)]
        self.cell_segments = [
            generate_probe_segment(CellSegment)]

    def test_defaults(self):
        self.assertEqual((5, 35), self.runner.p_denoise)
