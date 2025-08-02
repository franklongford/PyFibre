import logging

from traits.api import Instance

from pyfibre.io.object_io import (
    load_fibre_segments, load_cell_segments)
from pyfibre.utilities import log_time

from .metric_analysers import PLMetricAnalyser
from .shg_analyser import SHGAnalyser
from .shg_pl_trans_image import SHGPLTransImage


logger = logging.getLogger(__name__)


class SHGPLTransAnalyser(SHGAnalyser):

    multi_image = Instance(SHGPLTransImage)

    def _load_segments(self):
        """Load FibreSegment and CellSegment instances
        created during the analysis"""
        self._fibre_segments = load_fibre_segments(
            self._data_file, intensity_image=self.multi_image.shg_image)
        self._cell_segments = load_cell_segments(
            self._data_file, intensity_image=self.multi_image.pl_image)

    def _figures_kwargs(self):
        kwargs = super(SHGPLTransAnalyser, self)._figures_kwargs()

        kwargs['cell_regions'] = [cell.region for cell in self._cell_segments]

        return kwargs

    @log_time('PL METRICS')
    def create_metrics(self, sigma):
        """Perform metric analysis on segmented image

        Parameters
        ----------
        sigma: float

        Returns
        -------
        databases: list of pd.DataFrame
        """
        super(SHGPLTransAnalyser, self).create_metrics(sigma)

        logger.debug(" Performing PL Image analysis")

        metric_analyser = PLMetricAnalyser(
            filename=self.multi_image.name,
            image=self.multi_image.pl_image,
            sigma=sigma,
            segments=self._cell_segments
        )
        segment_metrics, global_metrics = metric_analyser.analyse()

        global_database = self._databases[0]
        global_database = global_database.append(
            global_metrics, ignore_index=False)

        logger.debug(" Cell segment analysis complete")

        self._databases = tuple(
            [global_database,
             self._databases[1],
             self._databases[2],
             segment_metrics]
        )
