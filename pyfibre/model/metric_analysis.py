import logging
import time
import numpy as np
import pandas as pd

from pyfibre.model.tools.analysis import (
    segment_analysis, fibre_network_analysis,
    cell_analysis
)
from pyfibre.model.tools.convertors import segments_to_binary, binary_to_segments
from pyfibre.model.tools.filters import form_structure_tensor
from pyfibre.utilities import flatten_list

logger = logging.getLogger(__name__)


class ImageAnalyser:

    def __init__(self, image, filename, objects, sigma):

        self.filename = filename
        self.image = image
        self.sigma = sigma

        self.objects = objects

        self.global_dataframe = None


class SHGAnalyser(ImageAnalyser):

    def __init__(self, image_shg, filename, fibre_networks, sigma):
        super().__init__(image=image_shg, filename=filename,
                         objects=fibre_networks, sigma=sigma)

        # Form structure tensors for each pixel
        self.shg_j_tensor = form_structure_tensor(
            image=self.image, sigma=self.sigma)

        self.fibre_dataframe = None
        self.global_metrics = None

    def analyse(self):

        "Analyse fibre network and individual regions"
        fibre_metrics = fibre_network_analysis(self.objects, self.image, self.sigma)

        fibre_filenames = pd.Series(
            ['{}_fibre_segment.pkl'.format(self.filename)] * len(self.objects),
            name='File')
        self.fibre_dataframe = pd.concat((fibre_filenames, fibre_metrics), axis=1)

        # Perform non-linear analysis on global region
        fibre_segments = [fibre_network.segment for fibre_network in self.objects]
        fibre_binary = segments_to_binary(fibre_segments, self.image.shape)
        global_segment = binary_to_segments(fibre_binary, self.image)[0]

        self.global_metrics = segment_analysis(global_segment, self.image, 'SHG Fibre')
        # Average linear properties over all regions
        self.global_averaging(self.global_metrics, fibre_metrics, fibre_binary)

    def global_averaging(self, global_fibre_metrics, fibre_metrics, fibre_binary):

        global_fibre_metrics['No. Fibres'] = sum([
            len(fibre_network.fibres) for fibre_network in self.objects])
        global_fibre_metrics['SHG Fibre Area'] = np.mean(fibre_metrics['Network Area'])
        global_fibre_metrics['SHG Fibre Coverage'] = np.sum(fibre_binary) / self.image.size
        global_fibre_metrics['SHG Fibre Eccentricity'] = np.mean(fibre_metrics['Network Eccentricity'])
        global_fibre_metrics['SHG Fibre Linearity'] = np.mean(fibre_metrics['Network Linearity'])
        global_fibre_metrics['SHG Fibre Density'] = np.mean(self.image[np.where(fibre_binary)])
        global_fibre_metrics['SHG Fibre Hu Moment 1'] = np.mean(fibre_metrics['Network Hu Moment 1'])
        global_fibre_metrics['SHG Fibre Hu Moment 2'] = np.mean(fibre_metrics['Network Hu Moment 2'])

        global_fibre_metrics = global_fibre_metrics.drop(['SHG Fibre Hu Moment 3', 'SHG Fibre Hu Moment 4'])

        global_fibre_metrics['SHG Fibre Waviness'] = np.nanmean(fibre_metrics['Mean Fibre Waviness'])
        global_fibre_metrics['SHG Fibre Length'] = np.nanmean(fibre_metrics['Mean Fibre Length'])
        global_fibre_metrics['SHG Fibre Cross-Link Density'] = np.nanmean(fibre_metrics['SHG Fibre Cross-Link Density'])

        logger.debug(fibre_metrics['SHG Network Degree'].values)

        global_fibre_metrics['SHG Fibre Network Degree'] = np.nanmean(fibre_metrics['SHG Network Degree'].values)
        global_fibre_metrics['SHG Fibre Network Eigenvalue'] = np.nanmean(fibre_metrics['SHG Network Eigenvalue'])
        global_fibre_metrics['SHG Fibre Network Connectivity'] = np.nanmean(
            fibre_metrics['SHG Network Connectivity'])


class PLAnalyser(ImageAnalyser):

    def __init__(self, image_pl, filename, cells, sigma):
        super().__init__(image=image_pl, filename=filename,
                         objects=cells, sigma=sigma)

        "Form structure tensors for each pixel"
        self.pl_j_tensor = form_structure_tensor(
            image=self.image, sigma=sigma)

        self.cell_dataframe = None
        self.global_metrics = None

    def analyse(self):

        cell_metrics = cell_analysis(self.objects, self.image)

        cell_filenames = pd.Series(
            ['{}_cell_segment.pkl'.format(self.filename)] * len(self.objects), name='File')
        self.cell_dataframe = pd.concat((cell_filenames, cell_metrics), axis=1)

        # Perform non-linear analysis on global region
        cell_segments = [cell.segment for cell in self.objects]
        cell_binary = segments_to_binary(cell_segments, self.image.shape)
        global_segment = binary_to_segments(cell_binary, self.image)[0]

        self.global_metrics = segment_analysis(global_segment, self.image, 'PL Cell')
        self.global_metrics = self.global_metrics.drop(
            ['PL Cell Hu Moment 3', 'PL Cell Hu Moment 4'])
        self.global_metrics['No. Cells'] = len(self.objects)


def metric_analysis(multi_image, filename, fibre_networks, cells, sigma,
                    shg_analysis=False, pl_analysis=False):

    global_dataframe = pd.Series()
    global_dataframe['File'] = '{}_global_segment.pkl'.format(filename)

    dataframes = [None, None]

    logger.debug(" Performing Image analysis")

    if shg_analysis:
        start = time.time()

        shg_analyser = SHGAnalyser(
            multi_image.shg_image, filename, fibre_networks, sigma
        )
        shg_analyser.analyse()

        dataframes[0] = shg_analyser.fibre_dataframe
        global_dataframe = pd.concat(
            (global_dataframe, shg_analyser.global_metrics), axis=0)

        end = time.time()
        logger.debug(f" Fibre segment analysis: {end-start} s")

    if pl_analysis:
        start = time.time()

        pl_analyser = PLAnalyser(
            multi_image.pl_image, filename, cells, sigma
        )

        pl_analyser.analyse()

        dataframes[1] = pl_analyser.cell_dataframe
        global_dataframe = pd.concat(
            (global_dataframe, pl_analyser.global_metrics), axis=0)
        end = time.time()

        logger.debug(f" Cell segment analysis: {end - start} s")

    return global_dataframe, dataframes
