import os
import time
import logging

import numpy as np
import pandas as pd

from pickle import UnpicklingError

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_closing

from skimage.measure import regionprops

import pyfibre.utilities as ut
from pyfibre.tools.multi_image import MultiLayerImage
import pyfibre.tools.analysis as an
import pyfibre.tools.segmentation as seg
from pyfibre.tools.extraction import network_extraction
from pyfibre.tools.filters import form_structure_tensor
from pyfibre.tools.figures import create_figure, create_tensor_image, create_region_image, create_network_image

logger = logging.getLogger(__name__)


def analyse_image(input_file_names, prefix, working_dir=None, scale=1.25,
                  p_intensity=(1, 99), p_denoise=(5, 35), sigma=0.5, alpha=0.5,
                  ow_metric=False, ow_segment=False, ow_network=False, ow_figure=False,
                  threads=8):
    """
    Analyse imput image by calculating metrics and sgenmenting via FIRE algorithm

    Parameters
    ----------

    input_file_name: str
        Full file path of image
    working_dir: str (optional)
        Working directory
    scale: float (optional)
        Unit of scale to resize image
    p_intensity: tuple (float); shape=(2,) (optional)
        Percentile range for intensity rescaling (used to remove outliers)
    p_denoise: tuple (float); shape=(2,) (optional)
        Parameters for non-linear means denoise algorithm (used to remove noise)
    sigma: float (optional)
        Standard deviation of Gaussian smoothing
    ow_metric: bool (optional)
        Force over-write of image metrics
    ow_network: bool (optional)
        Force over-write of image network
    threads: int (optional)
        Maximum number of threads to use for FIRE algorithm

    Returns
    -------

    metrics: array_like, shape=(11,)
        Calculated metrics for further analysis
    """

    cmap = 'viridis'
    if working_dir == None: working_dir = os.getcwd()

    data_dir = working_dir + '/data/'
    fig_dir = working_dir + '/fig/'

    if not os.path.exists(data_dir): os.mkdir(data_dir)
    if not os.path.exists(fig_dir): os.mkdir(fig_dir)

    image_name = prefix.split('/')[-1]
    filename = '{}'.format(data_dir + image_name)

    logger.info(f"Loading images for {prefix}")

    "Load and preprocess image"
    multi_image = MultiLayerImage(input_file_names, p_intensity)

    try:
        networks = ut.load_region(data_dir + image_name + "_network")
        networks_red = ut.load_region(data_dir + image_name + "_network_reduced")
        fibres = ut.load_region(data_dir + image_name + "_fibre")
    except (UnpicklingError, IOError, EOFError):
        logger.info("Cannot load networks for {}".format(image_name))
        ow_network = True

    try:
        fibre_seg = ut.load_region(data_dir + image_name + "_fibre_segment")
        if multi_image.pl_analysis: cell_seg = ut.load_region(data_dir + image_name + "_cell_segment")
    except (UnpicklingError, IOError, EOFError):
        logger.info("Cannot load segments for {}".format(image_name))
        ow_segment = True
        ow_metric = True
        ow_figure = True

    try:
        global_dataframe = pd.read_pickle('{}_global_metric.pkl'.format(data_dir + image_name))
        fibre_dataframe = pd.read_pickle('{}_fibre_metric.pkl'.format(data_dir + image_name))
        cell_dataframe = pd.read_pickle('{}_cell_metric.pkl'.format(data_dir + image_name))

    except (UnpicklingError, IOError, EOFError):
        logging("Cannot load metrics for {}".format(image_name))
        ow_metric = True

    logger.debug(f"Overwrite options:\n ow_network = {ow_network}\n ow_segment = {ow_segment}\
		\n ow_metric = {ow_metric}\n ow_figure = {ow_figure}")

    start = time.time()

    if ow_network:
        ow_segment = True

        start_net = time.time()

        (networks,
         networks_red,
         fibres) = network_extraction(
            multi_image.image_shg, data_dir + image_name,
            scale=scale, sigma=sigma, alpha=alpha, p_denoise=p_denoise,
            threads=threads)

        ut.save_region(networks, data_dir + image_name + "_network")
        ut.save_region(networks_red, data_dir + image_name + "_network_reduced")
        ut.save_region(fibres, data_dir + image_name + "_fibre")

        end_net = time.time()

        logger.debug(f"TOTAL NETWORK EXTRACTION TIME = {round(end_net - start_net, 3)} s")

    if ow_segment:

        ow_metric = True
        ow_figure = True

        start_seg = time.time()

        ow_metric = True
        ow_figure = True

        networks = ut.load_region(data_dir + image_name + "_network")
        networks_red = ut.load_region(data_dir + image_name + "_network_reduced")

        fibre_net_seg = seg.fibre_segmentation(
            multi_image.image_shg, networks, networks_red)

        if multi_image.pl_analysis:

            fibre_net_binary = seg.create_binary_image(
                fibre_net_seg, multi_image.shape)
            fibre_filter = np.where(fibre_net_binary, 2, 0.25)
            fibre_filter = gaussian_filter(fibre_filter, 1.0)

            cell_seg, fibre_col_seg = seg.cell_segmentation(
                multi_image.image_shg * fibre_filter,
                multi_image.image_pl, multi_image.image_tran, scale=scale)

            fibre_col_binary = seg.create_binary_image(fibre_col_seg, multi_image.shape)
            fibre_col_binary = binary_dilation(fibre_col_binary, iterations=2)
            fibre_col_binary = binary_closing(fibre_col_binary)

            fibre_binary = seg.mean_binary(
                multi_image.image_shg, fibre_net_binary, fibre_col_binary,
                min_size=150, min_intensity=0.13)

            fibre_seg = seg.get_segments(
                multi_image.image_shg, fibre_binary, 150, 0.05)
            cell_seg = seg.get_segments(
                multi_image.image_pl, ~fibre_binary, 250, 0.01)

            ut.save_region(cell_seg, '{}_cell_segment'.format(filename))
            ut.save_region(fibre_seg, '{}_fibre_segment'.format(filename))

        else:
            ut.save_region(fibre_net_seg, '{}_fibre_segment'.format(filename))

        end_seg = time.time()

        logger.debug(f"TOTAL SEGMENTATION TIME = {round(end_seg - start_seg, 3)} s")

    if ow_metric:

        start_met = time.time()

        "Load networks and segments"
        fibre_seg = ut.load_region(data_dir + image_name + "_fibre_segment")
        networks = ut.load_region(data_dir + image_name + "_network")
        networks_red = ut.load_region(data_dir + image_name + "_network_reduced")
        fibres = ut.load_region(data_dir + image_name + "_fibre")

        "Form nematic and structure tensors for each pixel"
        shg_j_tensor = form_structure_tensor(
            multi_image.image_shg, sigma=sigma)

        "Perform anisotropy analysis on each pixel"
        shg_pix_j_anis, shg_pix_j_angle, shg_pix_j_energy = an.tensor_analysis(shg_j_tensor)

        logger.debug("Performing fibre segment analysis")

        "Analyse fibre network"
        fibre_metrics = an.fibre_segment_analysis(
            multi_image.image_shg, networks, networks_red, fibres,
            fibre_seg, shg_j_tensor, shg_pix_j_anis, shg_pix_j_angle)

        fibre_filenames = pd.Series(['{}_fibre_segment.pkl'.format(filename)] * len(fibre_seg), name='File')

        if multi_image.pl_analysis:

            cell_seg = ut.load_region(data_dir + image_name + "_cell_segment")

            "Form nematic and structure tensors for each pixel"
            pl_j_tensor = form_structure_tensor(multi_image.image_pl, sigma=sigma)

            "Perform anisotropy analysis on each pixel"
            pl_pix_j_anis, pl_pix_j_angle, pl_pix_j_energy = an.tensor_analysis(pl_j_tensor)

            logger.debug("Performing cell segment analysis")

            muscle_metrics = an.cell_segment_analysis(
                multi_image.image_pl, fibre_seg, pl_j_tensor, pl_pix_j_anis, pl_pix_j_angle, 'Muscle')

            cell_metrics = an.cell_segment_analysis(
                multi_image.image_pl, cell_seg, pl_j_tensor, pl_pix_j_anis, pl_pix_j_angle)

            cell_filenames = pd.Series(['{}_cell_segment.pkl'.format(filename)] * len(cell_seg), name='File')
            cell_id = pd.Series(np.arange(len(cell_seg)), name='ID')

        else:
            cell_filenames = pd.Series(['{}_cell_segment.pkl'.format(filename)], name='File')
            cell_metrics = np.full_like(np.ones((1, len(cell_columns))), None)
            muscle_metrics = np.full_like(np.ones((len(fibre_seg), len(muscle_columns))), None)

        fibre_dataframe = pd.concat((fibre_filenames, fibre_metrics), axis=1)
        fibre_dataframe.to_pickle('{}_fibre_metric.pkl'.format(filename))

        cell_dataframe = pd.concat((cell_filenames, cell_metrics), axis=1)
        cell_dataframe.to_pickle('{}_cell_metric.pkl'.format(filename))

        muscle_dataframe = pd.concat((fibre_filenames, muscle_metrics), axis=1)
        muscle_dataframe.to_pickle('{}_muscle_metric.pkl'.format(filename))

        logger.debug("Performing Global Image analysis")

        fibre_binary = seg.create_binary_image(fibre_seg, multi_image.shape)
        global_binary = np.where(fibre_binary, 1, 0)
        global_segment = regionprops(global_binary, coordinates='xy')[0]

        global_fibre_metrics = an.segment_analysis(
            multi_image.image_shg, global_segment, shg_j_tensor, shg_pix_j_anis, shg_pix_j_angle,
            'SHG Fibre')

        global_fibre_metrics['SHG Fibre Area'] = np.mean(fibre_metrics['SHG Fibre Area'])
        global_fibre_metrics['SHG Fibre Coverage'] = np.sum(fibre_binary) / multi_image.size
        global_fibre_metrics['SHG Fibre Eccentricity'] = np.mean(fibre_metrics['SHG Fibre Eccentricity'])
        global_fibre_metrics['SHG Fibre Linearity'] = np.mean(fibre_metrics['SHG Fibre Linearity'])
        global_fibre_metrics['SHG Fibre Density'] = np.mean(multi_image.image_shg[np.where(fibre_binary)])
        global_fibre_metrics['SHG Fibre Hu Moment 1'] = np.mean(fibre_metrics['SHG Fibre Hu Moment 1'])
        global_fibre_metrics['SHG Fibre Hu Moment 2'] = np.mean(fibre_metrics['SHG Fibre Hu Moment 2'])
        global_fibre_metrics = global_fibre_metrics.drop(['SHG Fibre Hu Moment 3', 'SHG Fibre Hu Moment 4'])

        global_fibre_metrics['SHG Fibre Waviness'] = np.nanmean(fibre_metrics['SHG Fibre Waviness'])
        global_fibre_metrics['SHG Fibre Length'] = np.nanmean(fibre_metrics['SHG Fibre Length'])
        global_fibre_metrics['SHG Fibre Cross-Link Density'] = np.nanmean(fibre_metrics['SHG Fibre Cross-Link Density'])

        logger.debug(fibre_metrics['SHG Fibre Network Degree'].values)

        global_fibre_metrics['SHG Fibre Network Degree'] = np.nanmean(fibre_metrics['SHG Fibre Network Degree'].values)
        global_fibre_metrics['SHG Fibre Network Eigenvalue'] = np.nanmean(fibre_metrics['SHG Fibre Network Eigenvalue'])
        global_fibre_metrics['SHG Fibre Network Connectivity'] = np.nanmean(
            fibre_metrics['SHG Fibre Network Connectivity'])
        global_fibre_metrics['No. Fibres'] = len(ut.flatten_list(fibres))

        if multi_image.pl_analysis:

            global_muscle_metrics = an.segment_analysis(
                multi_image.image_pl, global_segment, shg_j_tensor,
                shg_pix_j_anis, shg_pix_j_angle, 'PL Muscle')

            global_muscle_metrics = global_muscle_metrics.drop(['PL Muscle Hu Moment 3', 'PL Muscle Hu Moment 4'])

            cell_binary = seg.create_binary_image(cell_seg, multi_image.shape)
            global_binary = np.where(cell_binary, 1, 0)
            global_segment = regionprops(global_binary, coordinates='xy')[0]

            global_cell_metrics = an.segment_analysis(
                multi_image.image_pl, global_segment, pl_j_tensor,
                pl_pix_j_anis, pl_pix_j_angle, 'PL Cell')

            global_cell_metrics = global_cell_metrics.drop(['PL Cell Hu Moment 3', 'PL Cell Hu Moment 4'])
            global_cell_metrics['No. Cells'] = len(cell_seg)

        else:
            global_cell_metrics = np.full_like(np.ones((len(['No. Cells'] + cell_columns[:-5]))), None)
            global_muscle_metrics = np.full_like(np.ones((len(muscle_columns))), None)

        global_metrics = pd.concat(
            (global_fibre_metrics, global_muscle_metrics, global_cell_metrics))
        filenames = pd.Series('{}_global_segment.pkl'.format(filename), name='File')

        global_dataframe = pd.concat((filenames, global_metrics), axis=1)
        global_dataframe.to_pickle('{}_global_metric.pkl'.format(filename))
        ut.save_region(global_segment, '{}_global_segment'.format(filename))

        end_met = time.time()

        logger.debug(f"TOTAL METRIC TIME = {round(end_met - start_met, 3)} s")

    if ow_figure:

        start_fig = time.time()

        networks = ut.load_region(data_dir + image_name + "_network")
        fibre_seg = ut.load_region(data_dir + image_name + "_fibre_segment")
        fibres = ut.load_region(data_dir + image_name + "_fibre")
        fibres = ut.flatten_list(fibres)

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
            cell_seg = ut.load_region(data_dir + image_name + "_cell_segment")
            cell_region_image = create_region_image(multi_image.image_pl, cell_seg)
            create_figure(multi_image.image_pl, fig_dir + image_name + '_PL', cmap='binary_r')
            create_figure(multi_image.image_tran, fig_dir + image_name + '_trans', cmap='binary_r')
            create_figure(cell_region_image, fig_dir + image_name + '_cell_seg')

        end_fig = time.time()

        logger.debug(f"TOTAL FIGURE TIME = {round(end_fig - start_fig, 3)} s")

    end = time.time()

    logger.info(f"TOTAL ANALYSIS TIME = {round(end - start, 3)} s")

    return global_dataframe, fibre_dataframe, cell_dataframe
