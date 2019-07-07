import logging
import numpy as np
import pandas as pd

from pyfibre.model.tools.analysis import (
    segment_analysis, fibre_segment_analysis,
    cell_segment_analysis
)
from pyfibre.model.tools import create_binary_image
from pyfibre.utilities import flatten_list
from pyfibre.model.tools import form_structure_tensor

logger = logging.getLogger(__name__)


class SHGAnalyser:

    def __init__(self, image_shg, filename, global_seg, fibre_seg,
                 networks, networks_red, fibres, sigma):

        self.filename = filename
        self.image_shg = image_shg
        "Form structure tensors for each pixel"
        self.shg_j_tensor = form_structure_tensor(
            image_shg, sigma=sigma)

        self.global_seg = global_seg[0]
        self.fibre_seg = fibre_seg
        self.networks = networks
        self.networks_red = networks_red
        self.fibres = fibres

        self.fibre_dataframe = None
        self.global_dataframe = None

    def analyse(self):

        "Analyse fibre network and individual regions"
        fibre_metrics = fibre_segment_analysis(
            self.image_shg, self.networks, self.networks_red, self.fibres,
            self.fibre_seg, self.shg_j_tensor)

        fibre_filenames = pd.Series(
            ['{}_fibre_segment.pkl'.format(self.filename)] * len(self.fibre_seg),
            name='File')
        self.fibre_dataframe = pd.concat((fibre_filenames, fibre_metrics), axis=1)

        "Perform non-linear analysis on global region"
        global_metrics = segment_analysis(
            self.image_shg, self.global_seg, self.shg_j_tensor, 'SHG Fibre')
        "Average linear properties over all regions"
        self.global_averaging(global_metrics, fibre_metrics)

        global_filenames = pd.Series('{}_global_segment.pkl'.format(self.filename), name='File')
        self.global_dataframe = pd.concat((global_filenames, global_metrics), axis=1)

    def global_averaging(self, global_fibre_metrics, fibre_metrics):

        fibre_binary = create_binary_image(self.fibre_seg, self.image_shg.shape)

        global_fibre_metrics['No. Fibres'] = len(flatten_list(self.fibres))
        global_fibre_metrics['SHG Fibre Area'] = np.mean(fibre_metrics['SHG Fibre Area'])
        global_fibre_metrics['SHG Fibre Coverage'] = np.sum(fibre_binary) / self.image_shg.size
        global_fibre_metrics['SHG Fibre Eccentricity'] = np.mean(fibre_metrics['SHG Fibre Eccentricity'])
        global_fibre_metrics['SHG Fibre Linearity'] = np.mean(fibre_metrics['SHG Fibre Linearity'])
        global_fibre_metrics['SHG Fibre Density'] = np.mean(self.image_shg[np.where(fibre_binary)])
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


class PLAnalyser:

    def __init__(self, image_pl, filename, global_seg, fibre_seg, cell_seg,
                 sigma):

        self.filename = filename
        self.image_pl = image_pl
        "Form structure tensors for each pixel"
        self.pl_j_tensor = form_structure_tensor(
            image_pl, sigma=sigma)

        self.global_seg = global_seg
        self.fibre_seg = fibre_seg
        self.cell_seg = cell_seg

        self.cell_dataframe = None
        self.global_dataframe = None

    def analyse(self):

        cell_metrics = cell_segment_analysis(
            self.image_pl, self.cell_seg, self.pl_j_tensor)

        cell_filenames = pd.Series(
            ['{}_cell_segment.pkl'.format(self.filename)] * len(self.cell_seg), name='File')
        self.cell_dataframe = pd.concat((cell_filenames, cell_metrics), axis=1)

        global_cell_metrics = segment_analysis(
            self.image_pl, self.global_seg[1], self.pl_j_tensor,
            'PL Cell')

        global_cell_metrics = global_cell_metrics.drop(['PL Cell Hu Moment 3', 'PL Cell Hu Moment 4'])
        global_cell_metrics['No. Cells'] = len(cell_seg)

        global_filenames = pd.Series('{}_global_segment.pkl'.format(self.filename), name='File')
        self.global_dataframe = pd.concat((global_filenames, global_metrics), axis=1)


def metric_analysis(multi_image, filename, global_seg, fibre_seg, cell_seg,
                    networks, networks_red, fibres, sigma):

    global_dataframe = pd.DataFrame()
    dataframes = [None, None]

    if multi_image.shg_analysis:
        logger.debug("Performing fibre segment analysis")

        shg_analyser = SHGAnalyser(
            multi_image.image_shg, filename, global_seg, fibre_seg,
            networks, networks_red, fibres, sigma
        )
        shg_analyser.analyse()

        dataframes[0] = shg_analyser.fibre_dataframe
        global_dataframe = pd.concat(
            (global_dataframe, shg_analyser.global_dataframe))

    if multi_image.pl_analysis:

        logger.debug("Performing cell segment analysis")

        pl_analyser = SHGAnalyser(
            multi_image.image_pl, filename, global_seg, fibre_seg,
            cell_seg, sigma
        )
        pl_analyser.analyse()

        dataframes[1] = pl_analyser.cell_dataframe
        global_dataframe = pd.concat(
            (global_dataframe, pl_analyser.global_dataframe))

    logger.debug("Performing Global Image analysis")

    return global_dataframe, dataframes
