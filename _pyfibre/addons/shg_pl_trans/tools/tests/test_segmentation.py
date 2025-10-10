from unittest import mock
import numpy as np

from pyfibre.tests.probe_classes.objects import ProbeFibreNetwork
from pyfibre.tests.pyfibre_test_case import PyFibreTestCase
from pyfibre.addons.shg_pl_trans.tests.probe_classes import (
    ProbeSHGPLTransImage)
from pyfibre.addons.shg_pl_trans.tests.fixtures import (
    test_fibre_mask)

from pyfibre.addons.shg_pl_trans.tools.segmentation import (
    shg_segmentation, shg_pl_trans_segmentation)


MODULE_PATH = "pyfibre.addons.shg_pl_trans.tools.segmentation"


class TestSegmentation(PyFibreTestCase):

    def setUp(self):
        self.fibre_mask = np.where(np.sum(test_fibre_mask, axis=0), 1, 0)
        self.multi_image = ProbeSHGPLTransImage()
        self.fibre_networks = [ProbeFibreNetwork()]

    def test_shg_segmentation(self):

        fibre_segments, cell_segments = shg_segmentation(
            self.multi_image, self.fibre_networks
        )

        self.assertEqual(0, len(fibre_segments))
        self.assertEqual(1, len(cell_segments))

        segment = cell_segments[0]
        self.assertAlmostEqual(
            120.6667, segment.region.intensity_image.max(), 4
        )

    @mock.patch(f'{MODULE_PATH}.rgb_segmentation')
    def test_shg_pl_trans_segmentation(self, mk_seg):
        mk_seg.return_value = self.fibre_mask, ~self.fibre_mask
        fibre_segments, cell_segments = shg_pl_trans_segmentation(
            self.multi_image, self.fibre_networks
        )
        self.assertEqual(0, len(fibre_segments))
        self.assertEqual(8, len(cell_segments))

        max_values = [64.6667, 79.3333, 61.6667, 106.3333,
                      89.6667, 116.0, 68.0, 101.3333]
        for max_value, segment in zip(max_values, cell_segments):
            self.assertAlmostEqual(
                max_value, segment.region.intensity_image.max(), 4
            )
