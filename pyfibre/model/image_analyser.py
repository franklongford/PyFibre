import os
import time
import logging

from pickle import UnpicklingError

from skimage.exposure import equalize_adapthist

from pyfibre.utilities import flatten_list
from pyfibre.model.tools.extraction import (
    build_network, fibre_network_assignment
)
from pyfibre.model.tools.figures import (
    create_figure, create_tensor_image, create_region_image,
    create_network_image
)
from pyfibre.model.tools.preprocessing import nl_means
from pyfibre.model.objects.multi_image import (
    MultiImage, SHGImage, SHGPLImage)
from pyfibre.io.object_io import (
    save_fibre_networks, load_fibre_networks,
    save_cells, load_cells)
from pyfibre.io.network_io import save_network, load_network
from pyfibre.io.database_io import save_database, load_database

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

        network = self.workflow.ow_network and isinstance(multi_image, SHGPLImage)
        segment = network or self.workflow.ow_segment
        metric = segment or self.workflow.ow_metric

        try:
            load_network(filename, "network")
        except Exception:
            logger.info("Cannot load networks for {}".format(filename))
            network = True
            segment = True
            metric = True

        try:
            load_fibre_networks(filename)
            if self.workflow.pl_analysis:
                load_cells(filename)
        except Exception:
            logger.info("Cannot load segments for {}".format(filename))
            segment = True
            metric = True

        try:
            load_database(filename, 'global_metric')
            load_database(filename, 'fibre_metric')
            load_database(filename, 'cell_metric')
        except (UnpicklingError, Exception):
            logger.info("Cannot load metrics for {}".format(filename))
            metric = True

        return network, segment, metric

    def _create_directory(self, prefix):

        (working_dir, data_dir, fig_dir,
         filename, figname) = self.get_filenames(prefix)

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

        image_nl = nl_means(image_equal, p_denoise=self.workflow.p_denoise)

        # Call FIRE algorithm to extract full image network
        logger.debug(
            f"Calling FIRE algorithm using image scale "
            f"{self.workflow.scale}  alpha  {self.workflow.alpha}"
        )
        network = build_network(
            image_nl,
            scale=self.workflow.scale,
            sigma=self.workflow.sigma,
            alpha=self.workflow.alpha)

        save_network(network, filename, "network")

        end_time = time.time()

        logger.info(
            f"TOTAL NETWORK EXTRACTION TIME = "
            f"{round(end_time - start_time, 3)} s")

        return network

    def segment_analysis(self, multi_image, filename):
        """Segment image into fiborous and cellular regions based on
        fibre network

        Parameters
        ----------
        multi_image: MultiImage
            MultiImage object to analyse
        filename: str
            Reference to original tif files for output data
        """

        start_time = time.time()

        network = load_network(filename, "network")

        fibre_networks = fibre_network_assignment(
            network, image=multi_image.shg_image)

        cells = multi_image.segmentation_algorithm(
            multi_image, fibre_networks,
            scale=self.workflow.scale
        )

        save_fibre_networks(fibre_networks, filename)
        save_cells(cells, filename, multi_image.shape)

        end_time = time.time()

        logger.info(f"TOTAL SEGMENTATION TIME = "
                    f"{round(end_time - start_time, 3)} s")

        return fibre_networks, cells

    def metric_analysis(self, multi_image, filename):
        """Perform metric analysis on segmented image

        Parameters
        ----------
        multi_image: MultiImage
            MultiImage object to analyse
        filename: str
            Reference to original tif files for output data
        """

        start_time = time.time()

        # Load networks and segments"
        fibre_networks = load_fibre_networks(
            filename, image=multi_image.shg_image)
        cells = load_cells(
            filename, image=multi_image.pl_image)

        global_dataframe, local_dataframes = generate_metrics(
            multi_image, filename, fibre_networks, cells,
            self.workflow.sigma,
            self.workflow.shg_analysis,
            self.workflow.pl_analysis)

        if self.workflow.shg_analysis:
            save_database(global_dataframe, filename, 'global_metric')
            save_database(local_dataframes[0], filename, 'fibre_metric')
        if self.workflow.pl_analysis:
            save_database(local_dataframes[1], filename, 'cell_metric')

        end_time = time.time()

        logger.info(f"TOTAL METRIC TIME = "
                    f"{round(end_time - start_time, 3)} s")

    def create_figures(self, multi_image, filename, figname):
        """Create and save figures"""

        start_fig = time.time()

        fibre_networks = load_fibre_networks(
            filename, image=multi_image.shg_image)

        regions = [fibre_network.region
                    for fibre_network in fibre_networks]
        fibres = [fibre_network.fibres
                  for fibre_network in fibre_networks]

        segment_graphs = [fibre_network.graph
                          for fibre_network in fibre_networks]
        fibre_graphs = [fibre.graph
                        for fibre in flatten_list(fibres)]

        if isinstance(multi_image, SHGImage):

            tensor_image = create_tensor_image(multi_image.shg_image)
            network_image = create_network_image(multi_image.shg_image, segment_graphs)
            fibre_image = create_network_image(multi_image.shg_image, fibre_graphs, 1)
            fibre_region_image = create_region_image(multi_image.shg_image, regions)

            create_figure(multi_image.shg_image, figname + '_SHG', cmap='binary_r')
            create_figure(tensor_image, figname + '_tensor')
            create_figure(network_image, figname + '_network')
            create_figure(fibre_image, figname + '_fibre')
            create_figure(fibre_region_image, figname + '_fibre_seg')

        if isinstance(multi_image, SHGPLImage):

            cells = load_cells(filename, image=multi_image.pl_image)

            cell_regions = [cell.region for cell in cells]
            cell_region_image = create_region_image(multi_image.pl_image, cell_regions)
            create_figure(cell_region_image, figname + '_cell_seg')

            if self.workflow.pl_analysis:
                create_figure(multi_image.pl_image, figname + '_PL', cmap='binary_r')
                create_figure(multi_image.trans_image, figname + '_trans', cmap='binary_r')

        end_fig = time.time()

        logger.info(f"TOTAL FIGURE TIME = {round(end_fig - start_fig, 3)} s")

    def get_filenames(self, prefix):

        image_name = os.path.basename(prefix)
        working_dir = (
            f"{os.path.dirname(prefix)}/{image_name}"
            "-pyfibre-analysis")

        data_dir = working_dir + '/data/'
        fig_dir = working_dir + '/fig/'

        filename = data_dir + image_name
        figname = fig_dir + image_name

        return working_dir, data_dir, fig_dir, filename, figname

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
        network, segment, metric = self.get_analysis_options(multi_image, filename)

        logger.debug(f"Analysis options:\n "
                     f"shg_analysis = {self.workflow.shg_analysis}\n "
                     f"pl_analysis = {self.workflow.pl_analysis}\n "
                     f"network_metrics = {network}\n "
                     f"segment_metrics = {segment}\n "
                     f"metric_analysis = {metric}\n "
                     f"save_figures = {self.workflow.save_figures}")

        start = time.time()

        if network:
            self.network_analysis(multi_image, filename)

        if segment:
            self.segment_analysis(multi_image, filename)

        if metric:
            self.metric_analysis(multi_image, filename)

        if self.workflow.save_figures:
            self.create_figures(multi_image, filename, figname)

        end = time.time()

        logger.info(f"TOTAL ANALYSIS TIME = {round(end - start, 3)} s")

        databases = ()

        if self.workflow.shg_analysis:
            global_dataframe = load_database(filename, 'global_metric')
            fibre_dataframe = load_database(filename, 'fibre_metric')
            databases += (global_dataframe, fibre_dataframe)

        if self.workflow.pl_analysis:
            cell_dataframe = load_database(filename, 'cell_metric')
            databases += (cell_dataframe,)

        return databases
