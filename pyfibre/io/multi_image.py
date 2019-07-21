import numpy as np

from traits.api import (
    HasTraits, ArrayOrNone, Property, Tuple, Bool,
    on_trait_change
)

from pyfibre.model.tools.preprocessing import clip_intensities


class MultiLayerImage(HasTraits):

    image_shg = ArrayOrNone()

    image_pl = ArrayOrNone()

    image_tran = ArrayOrNone()

    ow_network = Bool(False)

    ow_segment = Bool(False)

    ow_metric = Bool(False)

    ow_figure = Bool(False)

    shg_analysis = Bool(False)

    pl_analysis = Bool(False)

    p_intensity = Tuple((1, 99))

    shape = Property(Tuple, depends_on='image_shg')

    size = Property(Tuple, depends_on='image_shg')

    def _get_shape(self):
        if self.image_shg is not None:
            return self.image_shg.shape

    def _get_size(self):
        if self.image_shg is not None:
            return self.image_shg.size

    def preprocess_image_shg(self):

        self.shg_analysis = (
                self.shg_analysis
                and self.image_shg is not None
        )

        if self.image_shg is not None:
            self.image_shg = clip_intensities(
                self.image_shg, p_intensity=self.p_intensity
            )

    def preprocess_image_pl(self):

        self.pl_analysis = (
                self.pl_analysis
                and self.image_pl is not None
                and self.image_tran is not None
        )
