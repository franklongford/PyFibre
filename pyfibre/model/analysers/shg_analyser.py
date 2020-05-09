import os
from pickle import UnpicklingError
import logging
import time

import pandas as pd
from skimage.exposure import equalize_adapthist

from traits.api import Instance, List, Any, Tuple

from pyfibre.model.multi_image.shg_image import SHGImage
from pyfibre.model.objects.segments import (
    FibreSegment, CellSegment)
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.io.object_io import (
    save_fibre_networks, load_fibre_networks,
    save_fibre_segments, load_fibre_segments,
    save_cell_segments, load_cell_segments)
from pyfibre.io.network_io import save_network, load_network
from pyfibre.io.database_io import save_database, load_database
from pyfibre.model.analysers.metric_analyser import SHGMetricAnalyser
from pyfibre.model.tools.network_extraction import (
    build_network, fibre_network_assignment
)
from pyfibre.model.tools.preprocessing import nl_means
from pyfibre.utilities import flatten_list

from .base_analyser import BaseAnalyser

logger = logging.getLogger(__name__)


class SHGAnalyser(BaseAnalyser):

    multi_image = Instance(SHGImage)

    _network = Any

    _fibre_networks = List(FibreNetwork)

    _fibre_segments = List(FibreSegment)

    _cell_segments = List(CellSegment)

    _databases = Tuple()

    @property
    def data_path(self):
        return os.path.join(self.analysis_path, 'data')

    @property
    def fig_path(self):
        return os.path.join(self.analysis_path, 'fig')

    @property
    def _data_file(self):
        return os.path.join(self.data_path, self.multi_image.name)

    @property
    def _fig_file(self):
        return os.path.join(self.fig_path, self.multi_image.name)

    def _figures_kwargs(self):

        kwargs = {}

        fibres = [
            fibre_network.fibres for fibre_network in self._fibre_networks]
        kwargs['network_graphs'] = [
            fibre_network.graph for fibre_network in self._fibre_networks]
        kwargs['fibre_graphs'] = [
            fibre.graph for fibre in flatten_list(fibres)]
        kwargs['fibre_regions'] = [
            fibre_segment.region for fibre_segment in self._fibre_segments]
        kwargs['cell_regions'] = [cell.region for cell in self._cell_segments]

        return kwargs

    def _clear_attr(self):
        """Clears all private temporary attributes"""
        self._network = None
        self._fibre_networks = []
        self._fibre_segments = []
        self._cell_segments = []
        self._databases = (None, None, None, None)

    def _save_networks(self):
        """Save networkx Graphs representing fibre networks"""
        save_network(self._network, self._data_file, "network")
        save_fibre_networks(self._fibre_networks, self._data_file)

    def _load_networks(self):
        """Load networkx Graphs representing fibre network"""
        self._network = load_network(self._data_file, "network")
        self._fibre_networks = load_fibre_networks(self._data_file)

    def _save_segments(self):
        """Save FibreSegment and CellSegment instances
        created during the analysis"""
        save_fibre_segments(
            self._fibre_segments, self._data_file,
            shape=self.multi_image.shape)
        save_cell_segments(
            self._cell_segments, self._data_file,
            shape=self.multi_image.shape)

    def _load_segments(self):
        """Load FibreSegment and CellSegment instances
        created during the analysis"""
        self._fibre_segments = load_fibre_segments(
            self._data_file, intensity_image=self.multi_image.shg_image)
        self._cell_segments = load_cell_segments(
            self._data_file, intensity_image=self.multi_image.shg_image)

    def _save_databases(self):
        """Save pandas DataFrame instances created during the analysis"""
        save_database(self._databases[0], self._data_file, 'global_metric')
        save_database(self._databases[1], self._data_file, 'fibre_metric')
        save_database(self._databases[2], self._data_file, 'network_metric')
        save_database(self._databases[3], self._data_file, 'cell_metric')

    def _load_databases(self):
        """Load pandas DataFrame instances created during the analysis"""
        global_metrics = load_database(self._data_file, 'global_metric')
        fibre_metrics = load_database(self._data_file, 'fibre_metric')
        network_metrics = load_database(self._data_file, 'network_metric')
        cell_metrics = load_database(self._data_file, 'cell_metric')
        self._databases = tuple(
            [global_metrics, fibre_metrics,
             network_metrics, cell_metrics]
        )

    def make_directories(self):
        """Creates additional directories for analysis"""
        super(SHGAnalyser, self).make_directories()

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(self.fig_path):
            os.mkdir(self.fig_path)

    def get_analysis_options(self, workflow):
        """Get image-specific options for analysis"""

        network = workflow.ow_network
        segment = network or workflow.ow_segment
        metric = segment or workflow.ow_metric

        try:
            self._load_networks()
        except Exception:
            logger.info(
                f"Cannot load networks in {self.multi_image.name}")
            network = True
            segment = True
            metric = True

        try:
            self._load_segments()
        except Exception:
            logger.info(
                f"Cannot load segments in {self.multi_image.name}")
            segment = True
            metric = True

        try:
            self._load_databases()
        except (UnpicklingError, Exception):
            logger.info(
                f"Cannot load metrics for {self.multi_image.name}")
            metric = True

        return network, segment, metric

    def network_analysis(
            self, sigma, alpha, scale, p_denoise, fire_parameters):
        """Perform FIRE algorithm on image and save networkx
        objects for further analysis
        """

        logger.debug("Applying AHE to SHG image")
        image_equal = equalize_adapthist(self.multi_image.shg_image)

        logger.debug(
            "Performing NL Denoise using local windows {} {}".format(
                *p_denoise)
        )
        image_nl = nl_means(image_equal, p_denoise=p_denoise)

        # Call FIRE algorithm to extract full image network
        logger.debug(
            f"Calling FIRE algorithm using "
            f"image scale {scale}  "
            f"alpha  {alpha}"
        )
        self._network = build_network(
            image_nl,
            scale=scale,
            sigma=sigma,
            alpha=alpha,
            **fire_parameters)

        self._fibre_networks = fibre_network_assignment(self._network)

    def create_metrics(self, sigma):
        """Perform metric analysis on segmented image

        Parameters
        ----------
        sigma: float

        """

        global_dataframe = pd.Series(dtype=object)
        global_dataframe['File'] = self.multi_image.name

        start = time.time()

        logger.debug(" Performing SHG Image analysis")

        metric_analyser = SHGMetricAnalyser(
            filename=self.multi_image.name,
            image=self.multi_image.shg_image,
            sigma=sigma,
            networks=self._fibre_networks,
            segments=self._fibre_segments
        )

        (segment_merics,
         network_metrics,
         global_metrics) = metric_analyser.analyse()

        global_dataframe = global_dataframe.append(
            global_metrics, ignore_index=False)

        end = time.time()

        logger.debug(f" Fibre segment analysis: {end - start} s")

        self._databases = tuple(
            [global_dataframe, segment_merics, network_metrics, None]
        )

    def create_figures(self):
        """Create and save figures"""

        start_fig = time.time()

        self._load_segments()
        kwargs = self._figures_kwargs()

        self.multi_image.create_figures(
            self.multi_image,
            self._fig_file,
            **kwargs
        )

        end_fig = time.time()

        logger.info(
            f"TOTAL FIGURE TIME = "
            f"{round(end_fig - start_fig, 3)} s")

    def image_analysis(self, workflow):
        """
        Analyse input image by calculating metrics and
        segmenting via FIRE algorithm

        Returns
        -------
        databases: list of pd.DataFrame
        """

        network, segment, metric = self.get_analysis_options(
            workflow
        )

        self._clear_attr()

        start = time.time()

        self.multi_image.preprocess_images()

        # Load or create list of FibreNetwork instances
        if network:
            self.network_analysis(
                sigma=workflow.sigma,
                alpha=workflow.alpha,
                scale=workflow.scale,
                p_denoise=workflow.p_denoise,
                fire_parameters=workflow.fire_parameters)
            self._save_networks()
        else:
            self._load_fibre_networks()

        net_checkpoint = time.time()

        logger.info(
            f"TOTAL NETWORK EXTRACTION TIME = "
            f"{round(net_checkpoint - start, 3)} s")

        # Load or create lists of FibreSegments
        logger.debug("Segmenting Fibre and Cell regions")
        if segment:
            self_fibre_segments, self._cell_segments = (
                self.multi_image.segmentation_algorithm(
                    self._fibre_networks,
                    scale=workflow.scale
                )
            )
            self._save_segments()
        else:
            self._load_segments()

        seg_checkpoint = time.time()

        logger.info(
            f"TOTAL SEGMENTATION TIME = "
            f"{round(seg_checkpoint - net_checkpoint, 3)} s")

        if metric:
            self.create_metrics()
            self._save_databases()
        else:
            self._load_databases()

        end = time.time()

        logger.info(
            f"TOTAL METRIC TIME = "
            f"{round(end - seg_checkpoint, 3)} s")

        return self._databases
