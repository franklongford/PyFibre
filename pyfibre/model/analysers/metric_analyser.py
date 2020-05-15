from abc import ABC, abstractmethod
import logging
from functools import partial

import pandas as pd

from pyfibre.model.tools.metrics import (
    SHAPE_METRICS, STRUCTURE_METRICS, TEXTURE_METRICS,
    FIBRE_METRICS, NETWORK_METRICS, angle_analysis,
    fibre_network_metrics, segment_metrics
)
from pyfibre.utilities import flatten_list, nanmean

logger = logging.getLogger(__name__)


def metric_averaging(database, metrics, weights=None):
    """Create new pandas database from mean of selected
    metrics in existing database"""
    average_database = pd.Series(dtype=object)

    if weights is not None:
        if weights.size != database.shape[0]:
            raise ValueError(
                'Weights array must have same shape as '
                'database columns')

    for metric in metrics:
        try:
            average_database[metric] = nanmean(
                database[metric], weights)
        except KeyError:
            pass

    return average_database


class MetricAnalyser(ABC):

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

    def _global_averaging(self, local_metrics, segment_tag, image_tag,
                          weight_metric=None):

        shape_metrics = [
            f'{segment_tag} Segment {metric}'
            for metric in SHAPE_METRICS]
        texture_metrics = [
            f'{segment_tag} Segment {image_tag} {metric}'
            for metric in STRUCTURE_METRICS + TEXTURE_METRICS]
        fibre_metrics = [
            f'Mean {segment_tag} {metric}' for metric in
            FIBRE_METRICS]
        network_metrics = [
            f'{segment_tag} Network {metric}' for metric in
            NETWORK_METRICS]

        if weight_metric is None:
            weights = None
        else:
            weights = local_metrics[weight_metric].values

        global_metrics = metric_averaging(
            local_metrics,
            shape_metrics + texture_metrics +
            fibre_metrics + network_metrics,
            weights
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

    @abstractmethod
    def analyse(self):
        """Perform metric analysis, returning list of pandas
        DataFrame instances"""


class SHGMetricAnalyser(MetricAnalyser):

    def analyse(self):

        # Analyse fibre networks
        network_metrics = self._get_network_metrics()

        # Analyse fibre segments
        segment_metrics = self._get_segment_metrics(
            'SHG', 'fibre_segments.npy')

        # Average linear properties over all regions
        global_segment_metrics = self._global_averaging(
            segment_metrics, 'Fibre', 'SHG', 'Fibre Segment Area')
        global_network_metrics = self._global_averaging(
            network_metrics, 'Fibre', 'SHG')
        global_network_metrics['No. Fibres'] = sum(
            network_metrics['No. Fibres'])
        fibre_angles = flatten_list([
            [fibre.angle for fibre in fibre_network.fibres]
            for fibre_network in self.networks
        ])
        global_network_metrics['Fibre Angle SDI'], _ = angle_analysis(
            fibre_angles)
        global_metrics = pd.concat(
            (global_segment_metrics,
             global_network_metrics), axis=0)

        return segment_metrics, network_metrics, global_metrics


class PLMetricAnalyser(MetricAnalyser):

    def analyse(self):
        # Analyse individual cell regions
        segment_metrics = self._get_segment_metrics(
            'PL', 'cell_segment.npy')

        # Average linear properties over all regions
        global_metrics = self._global_averaging(
            segment_metrics, 'Cell', 'PL', 'Cell Segment Area')

        global_metrics['No. Cells'] = len(self.segments)

        return segment_metrics, global_metrics
