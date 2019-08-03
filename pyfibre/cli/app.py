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
from pyfibre.model.image_analysis import image_analysis
from pyfibre.io.database_io import save_database
from pyfibre.io.tif_reader import TIFReader
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
    '--shg', is_flag=True, default=False,
    help='Toggles analysis of SHG images'
)
@click.option(
    '--pl', is_flag=True, default=False,
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
    '--ow_figure', is_flag=True, default=False,
    help='Toggles overwrite figures'
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
    '--save_db', help='Output database filename',
    default=None
)
@click.argument(
    'file_path', type=click.Path(exists=True),
    required=False, default='.'
)
def pyfibre(file_path, key, sigma, alpha, save_db, debug,
        shg, pl, ow_metric, ow_segment, ow_network, ow_figure,
        test):

    if debug:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.DEBUG)
    else:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.INFO)

    logger = logging.getLogger(__name__)

    if test:
        file_path = os.path.dirname(
            os.path.dirname(__file__)) + '/tests/stubs'

    file_name, directory = parse_file_path(file_path)

    logger.info(ut.logo())
    logger.debug(f"{file_name} {directory}")

    input_files = parse_files(file_name, directory, key)
    reader = TIFReader(input_files,
                       shg=shg, pl=pl,
                       ow_network=ow_network,
                       ow_segment=ow_segment,
                       ow_metric=ow_metric,
                       ow_figure=ow_figure)
    reader.load_multi_images()

    global_database = pd.DataFrame()
    fibre_database = pd.DataFrame()
    cell_database = pd.DataFrame()

    for prefix, data in reader.files.items():

        databases = image_analysis(
            data['image'],
            prefix, sigma=sigma, alpha=alpha)

        global_database = global_database.append(databases[0], ignore_index=True)
        if shg:
            fibre_database = pd.concat([fibre_database, databases[1]])
        if pl:
            cell_database = pd.concat([cell_database, databases[2]])

        logger.debug(prefix)
        logger.debug("Global Image Analysis Metrics:")
        logger.debug(databases[0].iloc[0])

    if save_db is not None:
        save_database(global_database, save_db)
        if shg:
            save_database(fibre_database, save_db, 'fibre')
        if pl:
            save_database(cell_database, save_db, 'cell')
