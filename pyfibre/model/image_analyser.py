import os
import time
import logging

from pickle import UnpicklingError

from skimage.exposure import equalize_adapthist

from pyfibre.utilities import flatten_list
from pyfibre.model.tools.network_extraction import (
    build_network, fibre_network_assignment
)
from pyfibre.model.tools.preprocessing import nl_means
from pyfibre.model.multi_image.multi_images import SHGImage
from pyfibre.io.object_io import (
    save_fibre_networks, load_fibre_networks,
    save_fibre_segments, load_fibre_segments,
    save_cell_segments, load_cell_segments)
from pyfibre.io.network_io import save_network, load_network
from pyfibre.io.database_io import save_database, load_database
from pyfibre.io.utilities import get_file_names
from pyfibre.model.pyfibre_workflow import PyFibreWorkflow
from pyfibre.model.metric_analyser import generate_metrics

logger = logging.getLogger(__name__)


class ImageAnalyser:

    def __init__(self, workflow=None):
        """ Set parameters for ImageAnalyser routines

        Parameters
        ----------
        workflow: PyFibreWorkflow
            Instance containing information regarding PyFibre's
            Workflow
        """

        if workflow is not None:
            self.workflow = workflow
        else:
            self.workflow = PyFibreWorkflow()

    def get_analysis_options(self, multi_image, filename):
        """Get image-specific options"""

        type_check = isinstance(multi_image, SHGImage)

        network = type_check and self.workflow.ow_network
        segment = network or self.workflow.ow_segment
        metric = segment or self.workflow.ow_metric

        try:
            load_network(filename, "network")
            load_fibre_networks(filename)
        except Exception:
            logger.info(
                f"Cannot load networks for {filename}")
            network = True
            segment = True
            metric = True

        try:
            load_fibre_segments(filename)
            load_cell_segments(filename)
        except Exception:
            logger.info(
                f"Cannot load segments for {filename}")
            segment = True
            metric = True

        try:
            load_database(filename, 'global_metric')
            load_database(filename, 'fibre_metric')
            load_database(filename, 'network_metric')
            load_database(filename, 'cell_metric')
        except (UnpicklingError, Exception):
            logger.info(
                f"Cannot load metrics for {filename}")
            metric = True

        return network, segment, metric

    def _create_directory(self, prefix):

        (working_dir, data_dir, fig_dir,
         filename, figname) = get_file_names(prefix)

        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        return filename, figname

    def network_analysis(self, multi_image, filename):
        """Perform FIRE algorithm on image and save networkx
        object for further analysis

        Parameters
        ----------
        multi_image: MultiImage
            MultiImage object to analyse
        filename: str
            Reference to original tif files for output data
        """

        start_time = time.time()

        logger.debug("Applying AHE to SHG image")
        image_equal = equalize_adapthist(multi_image.shg_image)
        logger.debug(
            "Performing NL Denoise using local windows {} {}".format(
                *self.workflow.p_denoise)
        )

        image_nl = nl_means(
            image_equal, p_denoise=self.workflow.p_denoise)

        # Call FIRE algorithm to extract full image network
        logger.debug(
            f"Calling FIRE algorithm using "
            f"image scale {self.workflow.scale}  "
            f"alpha  {self.workflow.alpha}"
        )
        network = build_network(
            image_nl,
            scale=self.workflow.scale,
            sigma=self.workflow.sigma,
            alpha=self.workflow.alpha,
            **self.workflow.fire_parameters)

        save_network(network, filename, "network")

        fibre_networks = fibre_network_assignment(network)
        save_fibre_networks(fibre_networks, filename)

        end_time = time.time()

        logger.info(
            f"TOTAL NETWORK EXTRACTION TIME = "
            f"{round(end_time - start_time, 3)} s")

        return network, fibre_networks

    def segment_analysis(self, multi_image, filename, fibre_networks):
        """Segment image into fiborous and cellular regions based on
        fibre network

        Parameters
        ----------
        multi_image: MultiImage
            MultiImage object to analyse
        filename: str
            Reference to original tif files for output data
        fibre_networks: list of FibreNetwork
        """

        start_time = time.time()

        logger.debug("Segmenting Fibre and Cell regions")
        fibre_segments, cell_segments = multi_image.segmentation_algorithm(
            fibre_networks,
            scale=self.workflow.scale
        )

        save_fibre_segments(
            fibre_segments, filename, shape=multi_image.shape)
        save_cell_segments(
            cell_segments, filename, shape=multi_image.shape)

        end_time = time.time()

        logger.info(f"TOTAL SEGMENTATION TIME = "
                    f"{round(end_time - start_time, 3)} s")

        return fibre_segments, cell_segments

    def metric_analysis(self, multi_image, filename,
                        fibre_networks, fibre_segments, cell_segments):
        """Perform metric analysis on segmented image

        Parameters
        ----------
        multi_image: MultiImage
            MultiImage object to analyse
        filename: str
            Reference to original tif files for output data
        fibre_networks: list of FibreNetwork
        fibre_segments: list of FibreSegment
        cell_segments: list of CellSegment
        """

        start_time = time.time()

        global_dataframe, local_dataframes = generate_metrics(
            multi_image, filename, fibre_networks,
            fibre_segments, cell_segments,
            self.workflow.sigma)

        save_database(global_dataframe, filename, 'global_metric')
        save_database(local_dataframes[0], filename, 'fibre_metric')
        save_database(local_dataframes[1], filename, 'network_metric')
        save_database(local_dataframes[2], filename, 'cell_metric')

        end_time = time.time()

        logger.info(f"TOTAL METRIC TIME = "
                    f"{round(end_time - start_time, 3)} s")

        databases = tuple([global_dataframe] + local_dataframes)

        return databases

    def create_figures(self, multi_image, filename, figname):
        """Create and save figures"""

        start_fig = time.time()

        kwargs = {}

        fibre_networks = load_fibre_networks(filename)
        fibre_segments = load_fibre_segments(
            filename, intensity_image=multi_image.shg_image)

        fibres = [
            fibre_network.fibres for fibre_network in fibre_networks]
        kwargs['network_graphs'] = [
            fibre_network.graph for fibre_network in fibre_networks]
        kwargs['fibre_graphs'] = [
            fibre.graph for fibre in flatten_list(fibres)]
        kwargs['fibre_regions'] = [
            fibre_segment.region for fibre_segment in fibre_segments]

        try:
            cell_segments = load_cell_segments(
                filename, intensity_image=multi_image.pl_image)
            kwargs['cell_regions'] = [cell.region for cell in cell_segments]
        except IOError:
            pass

        multi_image.create_figures(figname, **kwargs)

        end_fig = time.time()

        logger.info(
            f"TOTAL FIGURE TIME = "
            f"{round(end_fig - start_fig, 3)} s")

    def image_analysis(self, multi_image, prefix):
        """
        Analyse input image by calculating metrics and
        segmenting via FIRE algorithm

        Parameters
        ----------
        multi_image: MultiImage
            MultiImage containing PL and SHG data
        prefix: str
           Prefix path of image

        Returns
        -------
        metrics: array_like, shape=(11,)
            Calculated metrics for further analysis
        """

        filename, figname = self._create_directory(prefix)
        network, segment, metric = self.get_analysis_options(
            multi_image, filename)

        logger.debug(f"Analysis options:\n "
                     f"Extract Network = {network}\n "
                     f"Segment Image = {segment}\n "
                     f"Generate Metrics = {metric}\n "
                     f"Save Figures = {self.workflow.save_figures}")

        start = time.time()

        # Load or create list of FibreNetwork instances
        if network:
            _, fibre_networks = self.network_analysis(
                multi_image, filename)
        else:
            fibre_networks = load_fibre_networks(filename)

        # Load or create lists of FibreSegments
        if segment:
            fibre_segments, cell_segments = self.segment_analysis(
                multi_image, filename, fibre_networks)
        else:
            fibre_segments = load_fibre_segments(
                filename, intensity_image=multi_image.shg_image)
            cell_segments = load_cell_segments(
                filename, intensity_image=multi_image.pl_image)

        # Load or create metric databases
        if metric:
            databases = self.metric_analysis(
                multi_image, filename, fibre_networks,
                fibre_segments, cell_segments)
        else:
            global_dataframe = load_database(filename, 'global_metric')
            fibre_dataframe = load_database(filename, 'fibre_metric')
            network_dataframe = load_database(filename, 'network_metric')
            cell_dataframe = load_database(filename, 'cell_metric')
            databases = (global_dataframe, fibre_dataframe,
                         network_dataframe, cell_dataframe)

        # Create figures
        if self.workflow.save_figures:
            self.create_figures(multi_image, filename, figname)

        end = time.time()

        logger.info(f"TOTAL ANALYSIS TIME = {round(end - start, 3)} s")

        return databases
