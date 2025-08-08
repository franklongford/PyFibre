import pandas as pd

from pyfibre.model.analysers.metric_analyser import MetricAnalyser
from pyfibre.model.tools.metrics import angle_analysis
from pyfibre.utilities import flatten_list


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

        # Overwrite a couple of metrics to calculate the sum
        # for the image, rather than the mean
        global_network_metrics['No. Fibres'] = sum(
            network_metrics['No. Fibres'])
        global_segment_metrics['Fibre Segment Coverage'] = sum(
            segment_metrics['Fibre Segment Area'] / self.image.size)

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
        global_metrics['Cell Segment Coverage'] = sum(
            segment_metrics['Cell Segment Area'] / self.image.size)

        return segment_metrics, global_metrics
