"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE 

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 18/02/2019
"""

import os
import logging
import click

import pandas as pd

import pyfibre.utilities as ut
from pyfibre.version import __version__
from pyfibre.model.image_analyser import ImageAnalyser
from pyfibre.io.database_io import save_database
from pyfibre.io.shg_pl_reader import collate_image_dictionary, SHGPLTransReader
from pyfibre.io.utils import parse_files, parse_file_path

import matplotlib
matplotlib.use("Agg")


@click.command()
@click.version_option(version=__version__)
@click.option(
    '--debug', is_flag=True, default=False,
    help="Prints extra debug information in pyfibre.log"
)
@click.option(
    '--shg_analysis', is_flag=True, default=False,
    help='Toggles analysis of SHG images'
)
@click.option(
    '--pl_analysis', is_flag=True, default=False,
    help='Toggles analysis of PL images'
)
@click.option(
    '--ow_metric', is_flag=True, default=False,
    help='Toggles overwrite analytic metrics'
)
@click.option(
    '--ow_segment', is_flag=True, default=False,
    help='Toggles overwrite image segmentation'
)
@click.option(
    '--ow_network', is_flag=True, default=False,
    help='Toggles overwrite network extraction'
)
@click.option(
    '--save_figures', is_flag=True, default=False,
    help='Toggles saving of figures'
)
@click.option(
    '--test', is_flag=True, default=False,
    help='Perform run on test image'
)
@click.option(
    '--key', help='Keywords to filter file names',
    default=None
)
@click.option(
    '--sigma', help='Gaussian smoothing standard deviation',
    default=0.5
)
@click.option(
    '--alpha', help='Alpha network coefficient',
    default=0.5
)
@click.option(
    '--database_name', help='Output database filename',
    default='pyfibre_database'
)
@click.option(
    '--log_name', help='Pyfibre log filename',
    default='pyfibre'
)
@click.argument(
    'file_path', type=click.Path(exists=True),
    required=False, default='.'
)
def pyfibre(file_path, key, sigma, alpha, log_name,
            database_name, debug,
            shg_analysis, pl_analysis, ow_metric, ow_segment,
            ow_network, save_figures, test):

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(filename=f"{log_name}.log", filemode="w",
                        level=level)

    logger = logging.getLogger(__name__)

    if test:
        file_path = os.path.dirname(
            os.path.dirname(__file__)) + '/tests/fixtures'
        shg_analysis = True
        pl_analysis = True

    file_name, directory = parse_file_path(file_path)

    logger.info(ut.logo())
    logger.debug(f"{file_name} {directory}")

    input_files = parse_files(file_name, directory, key)

    image_dictionary = collate_image_dictionary(input_files)

    reader = SHGPLTransReader()
    image_analyser = ImageAnalyser(
        sigma=sigma, alpha=alpha,
        shg_analysis=shg_analysis, pl_analysis=pl_analysis,
        ow_metric=ow_metric, ow_segment=ow_segment,
        ow_network=ow_network, save_figures=save_figures)
    global_database = pd.DataFrame()
    fibre_database = pd.DataFrame()
    cell_database = pd.DataFrame()

    for prefix, data in image_dictionary.items():

        logger.info(f"Processing image data for {data}")

        reader.assign_images(data)

        if 'PL-SHG' in data:
            reader.load_mode = 'PL-SHG File'
        elif 'PL' in data and 'SHG' in data:
            reader.load_mode = 'Separate Files'
        else:
            continue

        multi_image = reader.load_multi_image()

        databases = image_analyser.image_analysis(
            multi_image, prefix)

        global_database = global_database.append(
            databases[0], ignore_index=True)
        if shg_analysis:
            fibre_database = pd.concat([fibre_database, databases[1]])
        if pl_analysis:
            cell_database = pd.concat([cell_database, databases[2]])

        logger.debug(prefix)
        logger.debug("Global Image Analysis Metrics:")
        logger.debug(databases[0].iloc[0])

    save_database(global_database, database_name)
    if shg_analysis:
        save_database(fibre_database, database_name, 'fibre')
    if pl_analysis:
        save_database(cell_database, database_name, 'cell')
