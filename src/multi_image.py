import numpy as np

from preprocessing import load_shg_pl, clip_intensities


class MultiLayerImage():

    def __init__(self, file_path, p_intensity=(1, 99)):

        (self.image_shg,
         self.image_pl,
         self.image_tran) = load_shg_pl(file_path)

        self.shape = self.image_shg.shape
        self.size = self.image_shg.size

        self.shg_analysis = False
        self.pl_analysis = False
        self.check_analysis()

        self.p_intensity = p_intensity
        self.image_shg = clip_intensities(self.image_shg,
                                          p_intensity=self.p_intensity)
        if self.pl_analysis:
            self.image_pl = clip_intensities(
                self.image_pl, p_intensity=self.p_intensity)

    def check_analysis(self):

        self.shg_analysis = ~np.any(self.image_shg == None)
        self.pl_analysis = ~np.any(self.image_pl == None) * ~np.any(self.image_tran == None)
