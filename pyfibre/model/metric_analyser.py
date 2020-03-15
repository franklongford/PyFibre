import logging
import time

import numpy as np
import pandas as pd

from pyfibre.model.tools.metrics import (
    region_shape_metrics, fibre_network_metrics,
    segment_metrics, region_texture_metrics
)
from pyfibre.model.tools.convertors import (
    regions_to_binary, binary_to_regions)
from pyfibre.model.objects.multi_image import SHGPLImage


logger = logging.getLogger(__name__)


class MetricAnalyser:

    def __init__(
            self, image=None, filename=None, networks=None,
            segments=None, sigma=0.0001):

        self.filename = filename
        self.image = image
        self.networks = networks
        self.segments = segments
        self.sigma = sigma

        self.local_metrics = None
        self.global_metrics = None

    def _global_averaging(self, global_metrics, local_metrics, global_binary):

        global_metrics['No. Fibres'] = sum([
            len(fibre_network.fibres)
            for fibre_network in self.networks])
        global_metrics['SHG Fibre Area'] = np.mean(local_metrics['Network Area'])
        global_metrics['SHG Fibre Coverage'] = np.sum(global_binary) / self.image.size
        global_metrics['SHG Fibre Eccentricity'] = np.mean(local_metrics['Network Eccentricity'])
        global_metrics['SHG Fibre Linearity'] = np.mean(local_metrics['Network Linearity'])
        global_metrics['SHG Fibre Density'] = np.mean(self.image[np.where(global_binary)])
        global_metrics['SHG Fibre Hu Moment 1'] = np.mean(local_metrics['Network Hu Moment 1'])
        global_metrics['SHG Fibre Hu Moment 2'] = np.mean(local_metrics['Network Hu Moment 2'])

        global_metrics = global_metrics.drop(['SHG Fibre Hu Moment 3', 'SHG Fibre Hu Moment 4'])

        global_metrics['SHG Fibre Waviness'] = np.nanmean(local_metrics['Mean Fibre Waviness'])
        global_metrics['SHG Fibre Length'] = np.nanmean(local_metrics['Mean Fibre Length'])
        global_metrics['SHG Fibre Cross-Link Density'] = np.nanmean(local_metrics['SHG Fibre Cross-Link Density'])

        logger.debug(local_metrics['SHG Network Degree'].values)

        global_metrics['SHG Fibre Network Degree'] = np.nanmean(local_metrics['SHG Network Degree'].values)
        global_metrics['SHG Fibre Network Eigenvalue'] = np.nanmean(local_metrics['SHG Network Eigenvalue'])
        global_metrics['SHG Fibre Network Connectivity'] = np.nanmean(
            local_metrics['SHG Network Connectivity'])

    def _get_metrics(self, attr, metric_function, tag):

        logger.debug(f'Performing metrics for {tag}')

        # Analyse individual segments
        metrics = metric_function(
            attr, self.image, self.sigma)

        filenames = pd.Series(
            ['{}_{}'.format(self.filename, tag)] * len(attr),
            name='File')
        metrics = pd.concat((filenames, metrics), axis=1)

        return metrics

    def _get_network_metrics(self):
        return self._get_metrics(
            self.networks, fibre_network_metrics, 'fibre_networks.json')

    def _get_segment_metrics(self, tag):
        return self._get_metrics(
            self.segments, segment_metrics, tag)

    def _get_global_metrics(self, label):

        logger.debug(f'Performing global metrics for {label}')

        # Perform non-linear analysis on global region
        regions = [segment.region for segment in self.segments]
        global_binary = regions_to_binary(regions, self.image.shape)
        global_segment = binary_to_regions(global_binary, self.image)[0]

        global_metrics = region_shape_metrics(global_segment, label)
        global_metrics.append(
            region_texture_metrics(global_segment, self.image, label),
            ignore_index=False
        )

        return global_metrics, global_binary

    def analyse_shg(self):

        # Analyse fibre networks
        network_metrics = self._get_network_metrics()

        # Analyse fibre segments
        local_metrics = self._get_segment_metrics('fibre_segments.npy')

        local_metrics = pd.concat((network_metrics, local_metrics), axis=1)

        # Perform non-linear analysis on global region
        global_metrics, global_binary = self._get_global_metrics('SHG Fibre')

        # Average linear properties over all regions
        self._global_averaging(global_metrics, local_metrics, global_binary)

        return local_metrics, global_metrics

    def analyse_pl(self):

        # Analyse individual cell regions
        local_metrics = self._get_segment_metrics('cell_segment.npy')

        # Perform non-linear analysis on global region
        global_metrics, _ = self._get_global_metrics('PL Cell')

        global_metrics = global_metrics.drop(
            ['PL Cell Hu Moment 3', 'PL Cell Hu Moment 4'])
        global_metrics['No. Cells'] = len(self.segments)

        return local_metrics, global_metrics


def generate_metrics(
        multi_image, filename, fibre_networks,
        fibre_segments, cell_segments, sigma):

    global_dataframe = pd.Series(dtype=object)
    global_dataframe['File'] = '{}_global_segment.npy'.format(filename)

    local_dataframes = [None, None]

    logger.debug(" Performing SHG Image analysis")

    metric_analyser = MetricAnalyser(
        filename=filename, sigma=sigma
    )

    start = time.time()

    metric_analyser.image = multi_image.shg_image
    metric_analyser.networks = fibre_networks
    metric_analyser.segments = fibre_segments
    local_metrics, global_metrics = metric_analyser.analyse_shg()

    local_dataframes[0] = local_metrics
    global_dataframe = global_dataframe.append(
        global_metrics, ignore_index=False)

    end = time.time()
    logger.debug(f" Fibre segment analysis: {end-start} s")

    if isinstance(multi_image, SHGPLImage):

        logger.debug(" Performing PL Image analysis")

        start = time.time()

        metric_analyser.image = multi_image.pl_image
        metric_analyser.segments = cell_segments
        local_metrics, global_metrics = metric_analyser.analyse_pl()

        local_dataframes[1] = local_metrics
        global_dataframe = global_dataframe.append(
            global_metrics, ignore_index=False)

        end = time.time()
        logger.debug(f" Cell segment analysis: {end - start} s")

    return global_dataframe, local_dataframes
