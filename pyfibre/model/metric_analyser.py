import logging
import time
from functools import partial

import numpy as np
import pandas as pd

from pyfibre.model.tools.metrics import (
    SHAPE_METRICS, NEMATIC_METRICS, TEXTURE_METRICS,
    FIBRE_METRICS, NETWORK_METRICS, angle_analysis,
    fibre_network_metrics, segment_metrics
)
from pyfibre.model.multi_image.multi_images import SHGPLTransImage
from pyfibre.utilities import flatten_list

logger = logging.getLogger(__name__)


def metric_averaging(database, metrics):
    """Create new pandas database from mean of selected
    metrics in existing database"""
    average_database = pd.Series(dtype=object)

    for metric in metrics:
        try:
            average_database[metric] = np.nanmean(
                database[metric])
        except KeyError:
            pass

    return average_database


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

    def _global_averaging(self, local_metrics, segment_tag, image_tag):

        shape_metrics = [
            f'{segment_tag} Segment {metric}'
            for metric in SHAPE_METRICS]
        texture_metrics = [
            f'{segment_tag} Segment {image_tag} {metric}'
            for metric in NEMATIC_METRICS + TEXTURE_METRICS]
        fibre_metrics = [
            f'Mean {segment_tag} {metric}' for metric in
            FIBRE_METRICS]
        network_metrics = [
            f'{segment_tag} Network {metric}' for metric in
            NETWORK_METRICS]

        global_metrics = metric_averaging(
            local_metrics,
            shape_metrics + texture_metrics +
            fibre_metrics + network_metrics
        )

        return global_metrics

    def _get_metrics(self, attr, metric_function, tag):

        logger.debug(f'Performing metrics for {tag}')

        # Analyse individual segments
        metrics = metric_function(attr)

        filenames = pd.Series(
            ['{}_{}'.format(self.filename, tag)] * len(attr),
            name='File')
        metrics = pd.concat((filenames, metrics), axis=1)

        return metrics

    def _get_network_metrics(self):
        return self._get_metrics(
            self.networks, fibre_network_metrics, 'fibre_networks.json')

    def _get_segment_metrics(self, image_tag, tag):
        metric_func = partial(
            segment_metrics,
            image=self.image, image_tag=image_tag,
            sigma=self.sigma)
        return self._get_metrics(
            self.segments, metric_func, tag)

    def analyse_shg(self):

        # Analyse fibre networks
        network_metrics = self._get_network_metrics()

        # Analyse fibre segments
        segment_metrics = self._get_segment_metrics(
            'SHG', 'fibre_segments.npy')

        # Average linear properties over all regions
        global_segment_metrics = self._global_averaging(
            segment_metrics, 'Fibre', 'SHG')
        global_network_metrics = self._global_averaging(
            network_metrics, 'Fibre', 'SHG')
        global_network_metrics['No. Fibres'] = sum(
            network_metrics['No. Fibres'])
        fibre_angles = flatten_list([
            [fibre.angle for fibre in fibre_network.fibres]
            for fibre_network in self.networks
        ])
        global_network_metrics['Fibre Angle SDI'] = angle_analysis(
            fibre_angles)
        global_metrics = pd.concat(
            (global_segment_metrics,
             global_network_metrics), axis=0)

        return segment_metrics, network_metrics, global_metrics

    def analyse_pl(self):

        # Analyse individual cell regions
        segment_metrics = self._get_segment_metrics(
            'PL', 'cell_segment.npy')

        # Average linear properties over all regions
        global_metrics = self._global_averaging(
            segment_metrics, 'Cell', 'PL')

        global_metrics['No. Cells'] = len(self.segments)

        return segment_metrics, global_metrics


def generate_metrics(
        multi_image, filename, fibre_networks,
        fibre_segments, cell_segments, sigma):

    global_dataframe = pd.Series(dtype=object)
    global_dataframe['File'] = '{}_global_segment.npy'.format(filename)

    local_dataframes = [None, None, None]

    logger.debug(" Performing SHG Image analysis")

    metric_analyser = MetricAnalyser(
        filename=filename, sigma=sigma
    )

    start = time.time()

    metric_analyser.image = multi_image.shg_image
    metric_analyser.networks = fibre_networks
    metric_analyser.segments = fibre_segments
    (segment_merics,
     network_metrics,
     global_metrics) = metric_analyser.analyse_shg()

    local_dataframes[0] = segment_merics
    local_dataframes[1] = network_metrics
    global_dataframe = global_dataframe.append(
        global_metrics, ignore_index=False)

    end = time.time()
    logger.debug(f" Fibre segment analysis: {end-start} s")

    if isinstance(multi_image, SHGPLTransImage):

        logger.debug(" Performing PL Image analysis")

        start = time.time()

        metric_analyser.image = multi_image.pl_image
        metric_analyser.segments = cell_segments
        segment_merics, global_metrics = metric_analyser.analyse_pl()

        local_dataframes[2] = segment_merics
        global_dataframe = global_dataframe.append(
            global_metrics, ignore_index=False)

        end = time.time()
        logger.debug(f" Cell segment analysis: {end - start} s")

    return global_dataframe, local_dataframes
