import numpy as np

from pyfibre.model.tools.preprocessing import clip_intensities


class MultiLayerImage():

    def __init__(self, image_shg, image_pl, image_tran,
                 ow_network=False, ow_segment=False,
                 ow_metric=False, ow_figure=False,
                 shg_analysis=False, pl_analysis=False,
                 p_intensity=(1, 99)):

        self.image_shg = image_shg
        self.image_pl = image_pl
        self.image_tran = image_tran

        self.shape = self.image_shg.shape
        self.size = self.image_shg.size

        self.shg_analysis = shg_analysis
        self.pl_analysis = pl_analysis
        self.check_analysis()

        self.ow_network = ow_network
        self.ow_segment = ow_segment
        self.ow_metric = ow_metric
        self.ow_figure = ow_figure

        self.p_intensity = p_intensity
        self.image_shg = clip_intensities(self.image_shg,
                                          p_intensity=self.p_intensity)
        if self.pl_analysis:
            self.image_pl = clip_intensities(
                self.image_pl, p_intensity=self.p_intensity)

    def check_analysis(self):

        self.shg_analysis *= ~np.any(self.image_shg == None)
        self.pl_analysis *= ~np.any(self.image_pl == None) * ~np.any(self.image_tran == None)
