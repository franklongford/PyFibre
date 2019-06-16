import numpy as np

from pyfibre.tools.preprocessing import clip_intensities


class MultiLayerImage():

    def __init__(self, image_shg, image_pl, image_tran,
                 p_intensity=(1, 99)):

        self.image_shg = image_shg
        self.image_pl = image_pl
        self.image_tran = image_tran

        self.shape = self.image_shg.shape
        self.size = self.image_shg.size

        self.shg_analysis = False
        self.pl_analysis = False
        self.check_analysis()

        self.ow_network = False
        self.ow_segment = False
        self.ow_metric = False
        self.ow_figure = False

        self.p_intensity = p_intensity
        self.image_shg = clip_intensities(self.image_shg,
                                          p_intensity=self.p_intensity)
        if self.pl_analysis:
            self.image_pl = clip_intensities(
                self.image_pl, p_intensity=self.p_intensity)

    def check_analysis(self):

        self.shg_analysis = ~np.any(self.image_shg == None)
        self.pl_analysis = ~np.any(self.image_pl == None) * ~np.any(self.image_tran == None)
