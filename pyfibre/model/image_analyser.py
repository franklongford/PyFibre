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
from pyfibre.io.segment_io import load_segment, save_segment
from pyfibre.io.object_io import (
    save_fibre_networks, load_fibre_networks,
    save_objects, load_objects)
from pyfibre.io.network_io import save_network, load_network
from pyfibre.io.database_io import save_database, load_database

from pyfibre.model.metric_analysis import metric_analysis
from pyfibre.model.pyfibre_segmentation import cell_segmentation
from pyfibre.model.tools.convertors import segments_to_binary

logger = logging.getLogger(__name__)


class ImageAnalyser:

    def __init__(self, scale=1.25, p_denoise=(5, 35), sigma=0.5, alpha=0.5,
                 shg_analysis=True, pl_analysis=False, ow_metric=False,
                 ow_segment=False, ow_network=False, ow_figure=False):

        self.shg_analysis = shg_analysis
        self.pl_analysis = pl_analysis

        self.scale = scale
        self.p_denoise = p_denoise
        self.sigma = sigma
        self.alpha = alpha

        self.ow_metric = ow_metric
        self.ow_segment = ow_segment
        self.ow_network = ow_network
        self.ow_figure = ow_figure

    def get_ow_options(self, filename):

        ow_network = self.ow_network
        ow_segment = self.ow_segment
        ow_metric = self.ow_metric
        ow_figure = self.ow_figure

        try:
            load_network(filename, "network")
        except (UnpicklingError, Exception):
            logger.info("Cannot load networks for {}".format(filename))
            ow_network = True
            ow_segment = True
            ow_metric = True
            ow_figure = True

        try:
            load_segment(filename, "fibre_segment")
            if self.pl_analysis:
                load_segment(filename, "cell_segment")
        except (UnpicklingError, Exception):
            logger.info("Cannot load segments for {}".format(filename))
            ow_segment = True
            ow_metric = True
            ow_figure = True

        try:
            load_database(filename, 'global_metric')
            load_database(filename, 'fibre_metric')
            load_database(filename, 'cell_metric')
        except (UnpicklingError, Exception):
            logger.info("Cannot load metrics for {}".format(filename))
            ow_metric = True
            ow_figure = True

        return ow_network, ow_segment, ow_metric, ow_figure

    def network_analysis(self, multi_image, filename):

        start_net = time.time()

        logger.debug("Applying AHE to SHG image")
        image_equal = equalize_adapthist(multi_image.shg_image)
        logger.debug(
            "Performing NL Denoise using local windows {} {}".format(*self.p_denoise)
        )

        image_nl = nl_means(image_equal, p_denoise=self.p_denoise)

        # Call FIRE algorithm to extract full image network
        logger.debug(
            f"Calling FIRE algorithm using image scale {self.scale}  alpha  {self.alpha}"
        )
        network = build_network(image_nl, scale=self.scale,
                                sigma=self.sigma, alpha=self.alpha)

        save_network(network, filename, "network")

        end_net = time.time()

        logger.info(f"TOTAL NETWORK EXTRACTION TIME = {round(end_net - start_net, 3)} s")

        return network

    def segment_analysis(self, multi_image, filename):

        start_seg = time.time()

        network = load_network(filename, "network")

        fibre_networks = fibre_network_assignment(
            network, image=multi_image.shg_image)

        cells = cell_segmentation(
            multi_image, fibre_networks, scale=self.scale,
            pl_analysis=self.pl_analysis)

        end_seg = time.time()

        save_fibre_networks(fibre_networks, filename)
        save_objects(cells, filename, 'cells')

        logger.info(f"TOTAL SEGMENTATION TIME = {round(end_seg - start_seg, 3)} s")

        return fibre_networks, cells

    def metric_analysis(self, multi_image, filename):

        start_met = time.time()

        # Load networks and segments"
        fibre_networks = load_fibre_networks(filename)
        cells = load_objects(filename, "cells")

        global_dataframe, dataframes = metric_analysis(
            multi_image, filename, fibre_networks, cells,
            self.sigma, self.shg_analysis, self.pl_analysis)

        end_met = time.time()

        if self.shg_analysis:
            save_database(global_dataframe, filename, 'global_metric')
            save_database(dataframes[0], filename, 'fibre_metric')
        if self.pl_analysis:
            save_database(dataframes[1], filename, 'cell_metric')

        logger.info(f"TOTAL METRIC TIME = {round(end_met - start_met, 3)} s")

    def create_figures(self, multi_image, filename, figname):

        start_fig = time.time()

        fibre_networks = load_fibre_networks(filename)
        cells = load_objects(filename, "cells")

        fibre_segments = [fibre_network.segment for fibre_network in fibre_networks]
        networks = [fibre_network.graph for fibre_network in fibre_networks]
        fibres = [fibre_network.fibres for fibre_network in fibre_networks]
        fibres = [fibre.graph for fibre in flatten_list(fibres)]

        tensor_image = create_tensor_image(multi_image.shg_image)
        network_image = create_network_image(multi_image.shg_image, networks)
        fibre_image = create_network_image(multi_image.shg_image, fibres, 1)
        fibre_region_image = create_region_image(multi_image.shg_image, fibre_segments)

        create_figure(multi_image.shg_image, figname + '_SHG', cmap='binary_r')
        create_figure(tensor_image, figname + '_tensor')
        create_figure(network_image, figname + '_network')
        create_figure(fibre_image, figname + '_fibre')
        create_figure(fibre_region_image, figname + '_fibre_seg')

        if self.pl_analysis:
            cell_segments = [cell.segment for cell in cells]
            cell_region_image = create_region_image(multi_image.pl_image, cell_segments)
            create_figure(multi_image.pl_image, figname + '_PL', cmap='binary_r')
            create_figure(multi_image.trans_image, figname + '_trans', cmap='binary_r')
            create_figure(cell_region_image, figname + '_cell_seg')

        end_fig = time.time()

        logger.info(f"TOTAL FIGURE TIME = {round(end_fig - start_fig, 3)} s")

    def image_analysis(self, multi_image, prefix):
        """
        Analyse imput image by calculating metrics and segmenting via FIRE algorithm

        Parameters
        ----------
        multi_image: MultiImage
            MultiImage containing PL and SHG data
        prefix: str
           Prefix path of image
        scale: float (optional)
            Unit of scale to resize image
        p_denoise: tuple (float); shape=(2,) (optional)
            Parameters for non-linear means denoise algorithm (used to remove noise)
        sigma: float (optional)
            Standard deviation of Gaussian smoothing
        alpha: float (optional)
            Metric for hysterisis segmentation

        Returns
        -------
        metrics: array_like, shape=(11,)
            Calculated metrics for further analysis
        """

        working_dir = os.path.dirname(prefix)
        image_name = os.path.basename(prefix)
        data_dir = working_dir + '/data/'
        fig_dir = working_dir + '/fig/'

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        filename = data_dir + image_name
        figname = fig_dir + image_name
        ow_network, ow_segment, ow_metric, ow_figure = self.get_ow_options(filename)

        logger.debug(f"Overwrite options:\n "
                     f"shg_analysis = {self.shg_analysis}\n "
                     f"pl_analysis = {self.pl_analysis}\n "
                     f"ow_network = {ow_network}\n "
                     f"ow_segment = {ow_segment}\n "
                     f"ow_metric = {ow_metric}\n "
                     f"ow_figure = {ow_figure}")

        start = time.time()

        if ow_network:
            self.network_analysis(multi_image, filename)

        if ow_segment:
            self.segment_analysis(multi_image, filename)

        if ow_metric:
            self.metric_analysis(multi_image, filename)

        if ow_figure:
            self.create_figures(multi_image, filename, figname)

        end = time.time()

        logger.info(f"TOTAL ANALYSIS TIME = {round(end - start, 3)} s")

        databases = ()

        if self.shg_analysis:
            global_dataframe = load_database(filename, 'global_metric')
            fibre_dataframe = load_database(filename, 'fibre_metric')
            databases += (global_dataframe, fibre_dataframe)

        if self.pl_analysis:
            cell_dataframe = load_database(filename, 'cell_metric')
            databases += (cell_dataframe,)

        return databases
