import os
import time
import logging

from pickle import UnpicklingError

from pyfibre.utilities import flatten_list
from pyfibre.tools.extraction import network_extraction
from pyfibre.tools.figures import create_figure, create_tensor_image, create_region_image, create_network_image
from pyfibre.io.segment_io import load_segment, save_segment
from pyfibre.io.database_io import save_database, load_database

from pyfibre.pyfibre_metrics import metric_analysis
from pyfibre.pyfibre_segmentation import segment_image

logger = logging.getLogger(__name__)


def get_ow_options(multi_image, filename):

    try:
        load_segment(filename + "_network")
        load_segment(filename + "_network_reduced")
        load_segment(filename+ "_fibre")
    except (UnpicklingError, IOError, EOFError):
        logger.info("Cannot load networks for {}".format(filename))
        multi_image.ow_network = True
        multi_image.ow_segment = True
        multi_image.ow_metric = True
        multi_image.ow_figure = True

    try:
        load_segment(filename + "_fibre_segment")
        if multi_image.pl_analysis:
            load_segment(filename + "_cell_segment")
    except (UnpicklingError, IOError, EOFError):
        logger.info("Cannot load segments for {}".format(filename))
        multi_image.ow_segment = True
        multi_image.ow_metric = True
        multi_image.ow_figure = True

    try:
        load_database(filename, '_global_metric')
        load_database(filename, '_fibre_metric')
        load_database(filename, '_cell_metric')
    except (UnpicklingError, IOError, EOFError):
        logger.info("Cannot load metrics for {}".format(filename))
        multi_image.ow_metric = True


def analyse_image(multi_image, prefix, scale=1.25,
                  p_denoise=(5, 35), sigma=0.5, alpha=0.5, threads=8):
    """
    Analyse imput image by calculating metrics and sgenmenting via FIRE algorithm

    Parameters
    ----------
    multi_image: <class: MultiImage>
        MultiImage containing PL and SHG data
    prefix: str
       Prefix path of image
    scale: float (optional)
        Unit of scale to resize image
    p_denoise: tuple (float); shape=(2,) (optional)
        Parameters for non-linear means denoise algorithm (used to remove noise)
    sigma: float (optional)
        Standard deviation of Gaussian smoothing
    threads: int (optional)
        Maximum number of threads to use for FIRE algorithm

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
                 f"ow_network = {multi_image.ow_network}\n "
                 f"ow_segment = {multi_image.ow_segment}\n "
                 f"ow_metric = {multi_image.ow_metric}\n "
                 f"ow_figure = {multi_image.ow_figure}")

    start = time.time()

    if multi_image.ow_network:

        start_net = time.time()

        networks, networks_red, fibres = network_extraction(
            multi_image.image_shg, filename,
            scale=scale, sigma=sigma, alpha=alpha, p_denoise=p_denoise,
            threads=threads)

        save_segment(networks, filename + "_network")
        save_segment(networks_red, filename + "_network_reduced")
        save_segment(fibres, filename + "_fibre")

        end_net = time.time()

        logger.debug(f"TOTAL NETWORK EXTRACTION TIME = {round(end_net - start_net, 3)} s")

    if multi_image.ow_segment:

        start_seg = time.time()

        networks = load_segment(filename + "_network")
        networks_red = load_segment(filename + "_network_reduced")

        global_seg, fibre_seg, cell_seg = segment_image(
            multi_image, networks, networks_red, scale=scale)

        end_seg = time.time()

        save_segment(global_seg, '{}_global_segment'.format(filename))
        save_segment(fibre_seg, '{}_fibre_segment'.format(filename))
        save_segment(cell_seg, '{}_cell_segment'.format(filename))

        logger.debug(f"TOTAL SEGMENTATION TIME = {round(end_seg - start_seg, 3)} s")

    if multi_image.ow_metric:

        start_met = time.time()

        "Load networks and segments"
        global_seg = load_segment(filename + "_global_segment")
        fibre_seg = load_segment(filename + "_fibre_segment")
        cell_seg = load_segment(filename + "_cell_segment")
        networks = load_segment(filename + "_network")
        networks_red = load_segment(filename + "_network_reduced")
        fibres = load_segment(filename + "_fibre")

        (global_dataframe,
         fibre_dataframe,
         cell_dataframe,
         muscle_dataframe) = metric_analysis(
            multi_image, filename, global_seg, fibre_seg,
            cell_seg, networks, networks_red, fibres, sigma
        )

        end_met = time.time()

        save_database(global_dataframe, '{}_global_metric'.format(filename))
        save_database(fibre_dataframe, '{}_fibre_metric'.format(filename))
        save_database(cell_dataframe, '{}_cell_metric'.format(filename))
        save_database(muscle_dataframe, '{}_muscle_metric'.format(filename))

        logger.debug(f"TOTAL METRIC TIME = {round(end_met - start_met, 3)} s")

    if multi_image.ow_figure:

        start_fig = time.time()

        networks = load_segment(data_dir + image_name + "_network")
        fibre_seg = load_segment(data_dir + image_name + "_fibre_segment")
        fibres = load_segment(data_dir + image_name + "_fibre")
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
            cell_seg = load_segment(data_dir + image_name + "_cell_segment")
            cell_region_image = create_region_image(multi_image.image_pl, cell_seg)
            create_figure(multi_image.image_pl, fig_dir + image_name + '_PL', cmap='binary_r')
            create_figure(multi_image.image_tran, fig_dir + image_name + '_trans', cmap='binary_r')
            create_figure(cell_region_image, fig_dir + image_name + '_cell_seg')

        end_fig = time.time()

        logger.debug(f"TOTAL FIGURE TIME = {round(end_fig - start_fig, 3)} s")

    end = time.time()

    logger.info(f"TOTAL ANALYSIS TIME = {round(end - start, 3)} s")

    databases = ()

    if multi_image.shg_analysis:
        global_dataframe = load_database(filename, '_global_metric')
        fibre_dataframe = load_database(filename, '_fibre_metric')
        databases += (global_dataframe, fibre_dataframe)

    if multi_image.pl_analysis:
        cell_dataframe = load_database(filename, '_cell_metric')
        muscle_dataframe = load_database(filename, '_muscle_metric')
        databases += (cell_dataframe, muscle_dataframe)

    return databases
