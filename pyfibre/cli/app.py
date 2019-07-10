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

import matplotlib
matplotlib.use("Agg")


def parse_files(name, directory, key):

    input_files = []

    for file_name in name.split(','):
        if file_name.find('/') == -1:
            file_name = os.getcwd() + '/' + file_name
        input_files.append(file_name)

    if len(directory) != 0:
        for folder in directory.split(','):
            for file_name in os.listdir(folder):
                input_files += [folder + '/' + file_name]

    removed_files = []

    for key in key.split(','):
        for file_name in input_files:
            if ((file_name.find(key) == -1) and
                    (file_name not in removed_files)):
                removed_files.append(file_name)

    for file_name in removed_files:
        input_files.remove(file_name)

    return input_files


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
    '--directory', help='Directories to load tif files',
    type=click.Path(exists=True), default=None
)
@click.option(
    '--key', help='Keywords to filter file names',
    default=""
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
    'name', type=click.Path(exists=True),
    required=False, default=None
)
def run(name, directory, key, sigma, alpha, save_db, debug,
        shg, pl, ow_metric, ow_segment, ow_network, ow_figure, test):

    if debug is False:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.INFO)
    else:
        logging.basicConfig(filename="pyfibre.log", filemode="w",
                            level=logging.DEBUG)

    logger = logging.getLogger(__name__)

    if name is None:
        name = ""
    if directory is None:
        directory = ""
    if test:
        name = ""
        directory = os.path.dirname(
            os.path.dirname(__file__)) + '/tests/stubs'

    logger.info(ut.logo())
    logger.debug(f"{name} {directory}")

    input_files = parse_files(name, directory, key)
    reader = TIFReader(input_files, shg=shg, pl=pl,
                       ow_network=ow_network, ow_segment=ow_segment,
                       ow_metric=ow_metric, ow_figure=ow_figure)
    reader.load_multi_images()

    global_database = pd.DataFrame()
    fibre_database = pd.DataFrame()
    cell_database = pd.DataFrame()

    for prefix, data in reader.files.items():

        databases = image_analysis(
            data['image'],
            prefix, sigma=sigma, alpha=alpha)

        global_database = pd.concat([global_database, databases[0]])
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
