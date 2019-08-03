import os
import time
import logging

from pickle import UnpicklingError

from skimage.exposure import equalize_adapthist

from pyfibre.utilities import flatten_list
from pyfibre.model.tools.extraction import (
    FIRE, network_extraction
)
from pyfibre.model.tools.figures import (
    create_figure, create_tensor_image, create_region_image,
    create_network_image
)
from pyfibre.model.tools.preprocessing import nl_means
from pyfibre.io.segment_io import load_segment, save_segment
from pyfibre.io.network_io import (
    load_network, save_network, load_network_list, save_network_list
)
from pyfibre.io.database_io import save_database, load_database

from pyfibre.model.metric_analysis import metric_analysis
from pyfibre.model.pyfibre_segmentation import segment_image

logger = logging.getLogger(__name__)


def get_ow_options(multi_image, filename):

    try:
        load_network(filename, "network")
    except (UnpicklingError, Exception):
        logger.info("Cannot load networks for {}".format(filename))
        multi_image.ow_network = True
        multi_image.ow_segment = True
        multi_image.ow_metric = True
        multi_image.ow_figure = True

    try:
        load_segment(filename, "fibre_segment")
        if multi_image.pl_analysis:
            load_segment(filename, "cell_segment")
    except (UnpicklingError, Exception):
        logger.info("Cannot load segments for {}".format(filename))
        multi_image.ow_segment = True
        multi_image.ow_metric = True
        multi_image.ow_figure = True

    try:
        load_database(filename, 'global_metric')
        load_database(filename, 'fibre_metric')
        load_database(filename, 'cell_metric')
    except (UnpicklingError, Exception):
        logger.info("Cannot load metrics for {}".format(filename))
        multi_image.ow_metric = True


def image_analysis(multi_image, prefix, scale=1.25,
                   p_denoise=(5, 35), sigma=0.5, alpha=0.5):
    """
    Analyse imput image by calculating metrics and sgenmenting via FIRE algorithm

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
    get_ow_options(multi_image, filename)

    logger.debug(f"Overwrite options:\n "
                 f"shg_analysis = {multi_image.shg_analysis}\n "
                 f"pl_analysis = {multi_image.pl_analysis}\n "
                 f"ow_network = {multi_image.ow_network}\n "
                 f"ow_segment = {multi_image.ow_segment}\n "
                 f"ow_metric = {multi_image.ow_metric}\n "
                 f"ow_figure = {multi_image.ow_figure}")

    start = time.time()

    if multi_image.ow_network:

        start_net = time.time()

        logger.debug("Applying AHE to SHG image")
        image_equal = equalize_adapthist(multi_image.image_shg)
        logger.debug("Performing NL Denoise using local windows {} {}"
                     .format(*p_denoise))

        image_nl = nl_means(image_equal, p_denoise=p_denoise)

        "Call FIRE algorithm to extract full image network"
        logger.debug("Calling FIRE algorithm using image scale {}  alpha  {}".format(scale, alpha))
        network = FIRE(image_nl, scale=scale, sigma=sigma, alpha=alpha)

        save_network(network, filename, "network")

        networks, networks_red, fibres = network_extraction(network)

        save_network_list(networks, filename, "networks")
        save_network_list(networks_red, filename, "networks_red")
        save_network_list(fibres, filename, "fibres")

        end_net = time.time()

        logger.info(f"TOTAL NETWORK EXTRACTION TIME = {round(end_net - start_net, 3)} s")

    if multi_image.ow_segment:

        start_seg = time.time()

        network = load_network(filename, "network")
        try:
            networks = load_network_list(filename, "networks")
            networks_red = load_network_list(filename, "networks_red")
        except IOError:
            networks, networks_red, fibres = network_extraction(network)

            save_network_list(networks, filename, "networks")
            save_network_list(networks_red, filename, "networks_red")
            save_network_list(fibres, filename, "fibres")

        global_seg, fibre_seg, cell_seg = segment_image(
            multi_image, networks, networks_red, scale=scale)

        end_seg = time.time()

        save_segment(global_seg, filename, 'global_segment')
        save_segment(fibre_seg, filename, 'fibre_segment')
        save_segment(cell_seg, filename, 'cell_segment')

        logger.info(f"TOTAL SEGMENTATION TIME = {round(end_seg - start_seg, 3)} s")

    if multi_image.ow_metric:

        start_met = time.time()

        "Load networks and segments"
        global_seg = load_segment(filename, "global_segment")
        fibre_seg = load_segment(filename, "fibre_segment")
        cell_seg = load_segment(filename, "cell_segment")

        network = load_network(filename, "network")
        try:
            networks = load_network_list(filename, "networks")
            networks_red = load_network_list(filename, "networks_red")
            fibres = load_network_list(filename, "fibres")
        except IOError:
            networks, networks_red, fibres = network_extraction(network)

            save_network_list(networks, filename, "networks")
            save_network_list(networks_red, filename, "networks_red")
            save_network_list(fibres, filename, "fibres")

        global_dataframe, dataframes = metric_analysis(
            multi_image, filename, global_seg, fibre_seg,
            cell_seg, networks, networks_red, fibres, sigma
        )

        end_met = time.time()

        if multi_image.shg_analysis:
            save_database(global_dataframe, filename, 'global_metric')
            save_database(dataframes[0], filename, 'fibre_metric')
        if multi_image.pl_analysis:
            save_database(dataframes[1], filename, 'cell_metric')

        logger.info(f"TOTAL METRIC TIME = {round(end_met - start_met, 3)} s")

    if multi_image.ow_figure:

        start_fig = time.time()

        fibre_seg = load_segment(filename, "fibre_segment")
        network = load_network(filename, "network")
        networks, networks_red, fibres = network_extraction(network)

        fibres = flatten_list(fibres)

        tensor_image = create_tensor_image(multi_image.image_shg)
        network_image = create_network_image(multi_image.image_shg, networks)
        fibre_image = create_network_image(multi_image.image_shg, fibres, 1)
        fibre_region_image = create_region_image(multi_image.image_shg, fibre_seg)

        create_figure(multi_image.image_shg, fig_dir + image_name + '_SHG', cmap='binary_r')
        create_figure(tensor_image, fig_dir + image_name + '_tensor')
        create_figure(network_image, fig_dir + image_name + '_network')
        create_figure(fibre_image, fig_dir + image_name + '_fibre')
        create_figure(fibre_region_image, fig_dir + image_name + '_fibre_seg')

        if multi_image.pl_analysis:
            cell_seg = load_segment(filename, "cell_segment")
            cell_region_image = create_region_image(multi_image.image_pl, cell_seg)
            create_figure(multi_image.image_pl, fig_dir + image_name + '_PL', cmap='binary_r')
            create_figure(multi_image.image_tran, fig_dir + image_name + '_trans', cmap='binary_r')
            create_figure(cell_region_image, fig_dir + image_name + '_cell_seg')

        end_fig = time.time()

        logger.info(f"TOTAL FIGURE TIME = {round(end_fig - start_fig, 3)} s")

    end = time.time()

    logger.info(f"TOTAL ANALYSIS TIME = {round(end - start, 3)} s")

    databases = ()

    if multi_image.shg_analysis:
        global_dataframe = load_database(filename, 'global_metric')
        fibre_dataframe = load_database(filename, 'fibre_metric')
        databases += (global_dataframe, fibre_dataframe)

    if multi_image.pl_analysis:
        cell_dataframe = load_database(filename, 'cell_metric')
        databases += (cell_dataframe,)

    return databases
