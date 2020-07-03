from unittest import TestCase

import numpy as np


class PyFibreTestCase(TestCase):

    def assertArrayAlmostEqual(self, array1, array2, thresh=1E-8):
        return self.assertTrue(
            np.allclose(array1, array2, atol=thresh)
        )
