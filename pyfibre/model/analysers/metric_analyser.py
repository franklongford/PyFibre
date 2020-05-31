from abc import ABC, abstractmethod
import logging
from functools import partial

import pandas as pd

from pyfibre.model.tools.metrics import (
    SHAPE_METRICS, STRUCTURE_METRICS, TEXTURE_METRICS,
    FIBRE_METRICS, NETWORK_METRICS,
    fibre_network_metrics, segment_metrics
)
from pyfibre.utilities import nanmean

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
