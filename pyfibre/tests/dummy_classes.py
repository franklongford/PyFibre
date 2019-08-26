import numpy as np

from pyfibre.model.tools.extraction import Fibre


class DummyFibre(Fibre):

    def __init__(self, fibre_l=None, euclid_l=None, direction=None,
                 *args, **kwargs):

        if fibre_l is None or euclid_l is None:
            euclid_l = np.random.random_sample()
            fibre_l = euclid_l + np.random.random_sample()

        if direction is None:
            direction = [np.random.random_sample(),
                         np.random.random_sample()]

        super(DummyFibre, self).__init__(
            nodes=[], fibre_l=fibre_l, euclid_l=euclid_l,
            direction=direction, *args, **kwargs
        )