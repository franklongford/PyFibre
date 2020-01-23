import logging
import time
import numpy as np
import pandas as pd

from pyfibre.model.tools.metrics import (
    segment_shape_metrics, fibre_network_metrics,
    cell_metrics, segment_texture_metrics
)
from pyfibre.model.tools.convertors import segments_to_binary, binary_to_segments
from pyfibre.model.tools.filters import form_structure_tensor

logger = logging.getLogger(__name__)


class MetricAnalyser:

    def __init__(self, image, filename, objects, sigma):

        self.filename = filename
        self.image = image
        self.objects = objects
        self.sigma = sigma

        # Form structure tensors for each pixel
        self.j_tensor = form_structure_tensor(
            image=self.image, sigma=self.sigma)

        self.local_dataframe = None
        self.global_dataframe = None
        self.global_metrics = None


class SHGAnalyser(MetricAnalyser):

    def analyse(self):

        # Analyse fibre network and individual regions
        metrics = fibre_network_metrics(self.objects, self.image, self.sigma)

        fibre_filenames = pd.Series(
            ['{}_fibre_networks.json'.format(self.filename)] * len(self.objects),
            name='File')
        self.local_dataframe = pd.concat((fibre_filenames, metrics), axis=1)

        # Perform non-linear analysis on global region
        fibre_segments = [fibre_network.segment for fibre_network in self.objects]
        fibre_binary = segments_to_binary(fibre_segments, self.image.shape)
        global_segment = binary_to_segments(fibre_binary, self.image)[0]

        self.global_metrics = segment_shape_metrics(global_segment, 'SHG Fibre')
        self.global_metrics.append(
            segment_texture_metrics(global_segment, self.image, 'SHG Fibre'),
            ignore_index=False
        )

        # Average linear properties over all regions
        self.global_averaging(self.global_metrics, metrics, fibre_binary)

    def global_averaging(self, global_fibre_metrics, metrics, fibre_binary):

        global_fibre_metrics['No. Fibres'] = sum([
            len(fibre_network.fibres) for fibre_network in self.objects])
        global_fibre_metrics['SHG Fibre Area'] = np.mean(metrics['Network Area'])
        global_fibre_metrics['SHG Fibre Coverage'] = np.sum(fibre_binary) / self.image.size
        global_fibre_metrics['SHG Fibre Eccentricity'] = np.mean(metrics['Network Eccentricity'])
        global_fibre_metrics['SHG Fibre Linearity'] = np.mean(metrics['Network Linearity'])
        global_fibre_metrics['SHG Fibre Density'] = np.mean(self.image[np.where(fibre_binary)])
        global_fibre_metrics['SHG Fibre Hu Moment 1'] = np.mean(metrics['Network Hu Moment 1'])
        global_fibre_metrics['SHG Fibre Hu Moment 2'] = np.mean(metrics['Network Hu Moment 2'])

        global_fibre_metrics = global_fibre_metrics.drop(['SHG Fibre Hu Moment 3', 'SHG Fibre Hu Moment 4'])

        global_fibre_metrics['SHG Fibre Waviness'] = np.nanmean(metrics['Mean Fibre Waviness'])
        global_fibre_metrics['SHG Fibre Length'] = np.nanmean(metrics['Mean Fibre Length'])
        global_fibre_metrics['SHG Fibre Cross-Link Density'] = np.nanmean(metrics['SHG Fibre Cross-Link Density'])

        logger.debug(metrics['SHG Network Degree'].values)

        global_fibre_metrics['SHG Fibre Network Degree'] = np.nanmean(metrics['SHG Network Degree'].values)
        global_fibre_metrics['SHG Fibre Network Eigenvalue'] = np.nanmean(metrics['SHG Network Eigenvalue'])
        global_fibre_metrics['SHG Fibre Network Connectivity'] = np.nanmean(
            metrics['SHG Network Connectivity'])


class PLAnalyser(MetricAnalyser):

    def analyse(self):

        metrics = cell_metrics(self.objects, self.image)

        cell_filenames = pd.Series(
            ['{}_cell_segment.pkl'.format(self.filename)] * len(self.objects), name='File')
        self.local_dataframe = pd.concat((cell_filenames, metrics), axis=1)

        # Perform non-linear analysis on global region
        cell_segments = [cell.segment for cell in self.objects]
        cell_binary = segments_to_binary(cell_segments, self.image.shape)
        global_segment = binary_to_segments(cell_binary, self.image)[0]

        self.global_metrics = segment_shape_metrics(global_segment, 'PL Cell')
        self.global_metrics.append(
            segment_texture_metrics(global_segment, self.image, 'PL Cell'),
            ignore_index=False
        )

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
            multi_image.shg_image, filename,
            fibre_networks, sigma
        )
        shg_analyser.analyse()

        dataframes[0] = shg_analyser.local_dataframe
        global_dataframe = pd.concat(
            (global_dataframe, shg_analyser.global_metrics),
            axis=0)

        end = time.time()
        logger.debug(f" Fibre segment analysis: {end-start} s")

    if pl_analysis:
        start = time.time()

        pl_analyser = PLAnalyser(
            multi_image.pl_image, filename, cells, sigma
        )
        pl_analyser.analyse()

        dataframes[1] = pl_analyser.local_dataframe
        global_dataframe = pd.concat(
            (global_dataframe, pl_analyser.global_metrics), axis=0)
        end = time.time()

        logger.debug(f" Cell segment analysis: {end - start} s")

    return global_dataframe, dataframes
