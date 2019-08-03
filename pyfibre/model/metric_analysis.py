import logging
import time
import numpy as np
import pandas as pd

from pyfibre.model.tools.analysis import (
    segment_analysis, fibre_segment_analysis,
    cell_segment_analysis
)
from pyfibre.model.tools.segmentation import create_binary_image
from pyfibre.model.tools.filters import form_structure_tensor
from pyfibre.utilities import flatten_list

logger = logging.getLogger(__name__)


class ImageAnalyser:

    def __init__(self, image, filename, global_seg, fibre_seg, sigma):

        self.filename = filename
        self.image = image
        self.sigma = sigma

        self.global_seg = global_seg
        self.fibre_seg = fibre_seg

        self.global_dataframe = None


class SHGAnalyser(ImageAnalyser):

    def __init__(self, image_shg, filename, global_seg, fibre_seg,
                 networks, networks_red, fibres, sigma):
        super().__init__(image=image_shg, filename=filename,
                         global_seg=global_seg, fibre_seg=fibre_seg,
                         sigma=sigma)

        "Form structure tensors for each pixel"
        self.shg_j_tensor = form_structure_tensor(
            image=self.image, sigma=self.sigma)

        self.networks = networks
        self.networks_red = networks_red
        self.fibres = fibres

        self.fibre_dataframe = None
        self.global_metrics = None

    def analyse(self):

        "Analyse fibre network and individual regions"
        fibre_metrics = fibre_segment_analysis(
            self.image, self.networks, self.networks_red, self.fibres,
            self.fibre_seg, self.shg_j_tensor)

        fibre_filenames = pd.Series(
            ['{}_fibre_segment.pkl'.format(self.filename)] * len(self.fibre_seg),
            name='File')
        self.fibre_dataframe = pd.concat((fibre_filenames, fibre_metrics), axis=1)

        "Perform non-linear analysis on global region"
        self.global_metrics = segment_analysis(
            self.image, self.global_seg, self.shg_j_tensor, 'SHG Fibre')
        "Average linear properties over all regions"
        self.global_averaging(self.global_metrics, fibre_metrics)

    def global_averaging(self, global_fibre_metrics, fibre_metrics):

        fibre_binary = create_binary_image(self.fibre_seg, self.image.shape)

        global_fibre_metrics['No. Fibres'] = len(flatten_list(self.fibres))
        global_fibre_metrics['SHG Fibre Area'] = np.mean(fibre_metrics['SHG Fibre Area'])
        global_fibre_metrics['SHG Fibre Coverage'] = np.sum(fibre_binary) / self.image.size
        global_fibre_metrics['SHG Fibre Eccentricity'] = np.mean(fibre_metrics['SHG Fibre Eccentricity'])
        global_fibre_metrics['SHG Fibre Linearity'] = np.mean(fibre_metrics['SHG Fibre Linearity'])
        global_fibre_metrics['SHG Fibre Density'] = np.mean(self.image[np.where(fibre_binary)])
        global_fibre_metrics['SHG Fibre Hu Moment 1'] = np.mean(fibre_metrics['SHG Fibre Hu Moment 1'])
        global_fibre_metrics['SHG Fibre Hu Moment 2'] = np.mean(fibre_metrics['SHG Fibre Hu Moment 2'])

        global_fibre_metrics = global_fibre_metrics.drop(['SHG Fibre Hu Moment 3', 'SHG Fibre Hu Moment 4'])

        global_fibre_metrics['SHG Fibre Waviness'] = np.nanmean(fibre_metrics['SHG Fibre Waviness'])
        global_fibre_metrics['SHG Fibre Length'] = np.nanmean(fibre_metrics['SHG Fibre Length'])
        global_fibre_metrics['SHG Fibre Cross-Link Density'] = np.nanmean(fibre_metrics['SHG Fibre Cross-Link Density'])

        logger.debug(fibre_metrics['SHG Fibre Network Degree'].values)

        global_fibre_metrics['SHG Fibre Network Degree'] = np.nanmean(fibre_metrics['SHG Fibre Network Degree'].values)
        global_fibre_metrics['SHG Fibre Network Eigenvalue'] = np.nanmean(fibre_metrics['SHG Fibre Network Eigenvalue'])
        global_fibre_metrics['SHG Fibre Network Connectivity'] = np.nanmean(
            fibre_metrics['SHG Fibre Network Connectivity'])


class PLAnalyser(ImageAnalyser):

    def __init__(self, image_pl, filename, global_seg, fibre_seg, cell_seg,
                 sigma):
        super().__init__(image=image_pl, filename=filename,
                         global_seg=global_seg, fibre_seg=fibre_seg,
                         sigma=sigma)

        "Form structure tensors for each pixel"
        self.pl_j_tensor = form_structure_tensor(
            image=self.image, sigma=sigma)

        self.cell_seg = cell_seg

        self.cell_dataframe = None
        self.global_metrics = None

    def analyse(self):

        cell_metrics = cell_segment_analysis(
            self.image, self.cell_seg, self.pl_j_tensor)

        cell_filenames = pd.Series(
            ['{}_cell_segment.pkl'.format(self.filename)] * len(self.cell_seg), name='File')
        self.cell_dataframe = pd.concat((cell_filenames, cell_metrics), axis=1)

        self.global_metrics = segment_analysis(
            self.image, self.global_seg, self.pl_j_tensor,
            'PL Cell')

        self.global_metrics = self.global_metrics.drop(
            ['PL Cell Hu Moment 3', 'PL Cell Hu Moment 4'])
        self.global_metrics['No. Cells'] = len(self.cell_seg)


def metric_analysis(multi_image, filename, global_seg, fibre_seg, cell_seg,
                    networks, networks_red, fibres, sigma):

    global_dataframe = pd.Series()
    global_dataframe['File'] = '{}_global_segment.pkl'.format(filename)

    dataframes = [None, None]

    logger.debug(" Performing Image analysis")

    if multi_image.shg_analysis:
        start = time.time()

        shg_analyser = SHGAnalyser(
            multi_image.image_shg, filename, global_seg[0], fibre_seg,
            networks, networks_red, fibres, sigma
        )
        shg_analyser.analyse()

        dataframes[0] = shg_analyser.fibre_dataframe
        global_dataframe = pd.concat(
            (global_dataframe, shg_analyser.global_metrics), axis=0)

        end = time.time()
        logger.debug(f" Fibre segment analysis: {end-start} s")

    if multi_image.pl_analysis:
        start = time.time()

        pl_analyser = PLAnalyser(
            multi_image.image_pl, filename, global_seg[1], fibre_seg,
            cell_seg, sigma
        )

        pl_analyser.analyse()

        dataframes[1] = pl_analyser.cell_dataframe
        global_dataframe = pd.concat(
            (global_dataframe, pl_analyser.global_metrics), axis=0)
        end = time.time()

        logger.debug(f" Cell segment analysis: {end - start} s")

    return global_dataframe, dataframes
