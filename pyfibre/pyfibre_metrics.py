import logging
import numpy as np
import pandas as pd

from pyfibre.tools.analysis import (
    segment_analysis, fibre_segment_analysis,
    cell_segment_analysis
)
from pyfibre.tools.segmentation import create_binary_image
from pyfibre.utilities import flatten_list
from pyfibre.tools.filters import form_structure_tensor

logger = logging.getLogger(__name__)


def global_averaging(multi_image, global_fibre_metrics, fibre_metrics,
                     fibre_seg, fibres):

    fibre_binary = create_binary_image(fibre_seg, multi_image.shape)

    global_fibre_metrics['No. Fibres'] = len(flatten_list(fibres))
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


def metric_analysis(multi_image, filename, global_seg, fibre_seg, cell_seg,
                    networks, networks_red, fibres, sigma)

    "Form nematic and structure tensors for each pixel"
    shg_j_tensor = form_structure_tensor(
        multi_image.image_shg, sigma=sigma)

    logger.debug("Performing fibre segment analysis")

    "Analyse fibre network"
    fibre_metrics = fibre_segment_analysis(
        multi_image.image_shg, networks, networks_red, fibres,
        fibre_seg, shg_j_tensor)

    fibre_filenames = pd.Series(['{}_fibre_segment.pkl'.format(filename)] * len(fibre_seg), name='File')
    fibre_dataframe = pd.concat((fibre_filenames, fibre_metrics), axis=1)

    global_fibre_metrics = segment_analysis(
        multi_image.image_shg, global_seg[0], shg_j_tensor, 'SHG Fibre')
    global_averaging(multi_image, global_fibre_metrics, fibre_metrics, fibre_seg, fibres)


    if multi_image.pl_analysis:

        "Form nematic and structure tensors for each pixel"
        pl_j_tensor = form_structure_tensor(multi_image.image_pl, sigma=sigma)

        logger.debug("Performing cell segment analysis")

        muscle_metrics = cell_segment_analysis(
            multi_image.image_pl, fibre_seg, pl_j_tensor, 'Muscle')

        cell_metrics = cell_segment_analysis(
            multi_image.image_pl, cell_seg, pl_j_tensor)

        cell_filenames = pd.Series(['{}_cell_segment.pkl'.format(filename)] * len(cell_seg), name='File')

        global_muscle_metrics = segment_analysis(
            multi_image.image_pl, global_segment, shg_j_tensor,
            'PL Muscle')

        global_muscle_metrics = global_muscle_metrics.drop(['PL Muscle Hu Moment 3', 'PL Muscle Hu Moment 4'])

        global_cell_metrics = segment_analysis(
            multi_image.image_pl, global_segment, pl_j_tensor,
            'PL Cell')

        global_cell_metrics = global_cell_metrics.drop(['PL Cell Hu Moment 3', 'PL Cell Hu Moment 4'])
        global_cell_metrics['No. Cells'] = len(cell_seg)

    else:
        cell_filenames = pd.Series(['{}_cell_segment.pkl'.format(filename)], name='File')
        cell_metrics = np.full_like(np.ones((1, 10)), None)
        muscle_metrics = np.full_like(np.ones((fibre_metrics.shape[1], 10)), None)

        global_cell_metrics = np.full_like(np.ones((len(['No. Cells'] + cell_columns[:-5]))), None)
        global_muscle_metrics = np.full_like(np.ones((len(muscle_columns))), None)

    cell_dataframe = pd.concat((cell_filenames, cell_metrics), axis=1)
    muscle_dataframe = pd.concat((cell_filenames, muscle_metrics), axis=1)

    logger.debug("Performing Global Image analysis")

    global_metrics = pd.concat(
        (global_fibre_metrics, global_muscle_metrics, global_cell_metrics))
    global_filenames = pd.Series('{}_global_segment.pkl'.format(filename), name='File')

    global_dataframe = pd.concat((global_filenames, global_metrics), axis=1)

    return global_dataframe, fibre_dataframe, cell_dataframe, muscle_dataframe