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
from pyfibre.pyfibre_analyse_image import analyse_image
from pyfibre.io.database_writer import write_database
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
@click.option(
    '--threads', help='Number of threads per processor',
    default=8
)
@click.argument(
    'name', type=click.Path(exists=True),
    required=False, default=None
)
def run(name, directory, key, sigma, alpha, save_db, threads, debug,
        ow_metric, ow_segment, ow_network, ow_figure):

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

    logger.info(ut.logo())
    logger.debug(f"{name} {directory}")

    input_files = parse_files(name, directory, key)
    reader = TIFReader(input_files, ow_network, ow_segment,
                       ow_metric, ow_figure)
    reader.load_multi_images()

    cell_database = pd.DataFrame()
    fibre_database = pd.DataFrame()
    global_database = pd.DataFrame()

    for prefix, data in enumerate(reader.files):

        (data_global,
         data_segment,
         data_cell) = analyse_image(
            data['image'],
            prefix, sigma=sigma,
            threads=threads, alpha=alpha)

        global_database = pd.concat([global_database, data_global])
        fibre_database = pd.concat([fibre_database, data_segment])
        cell_database = pd.concat([cell_database, data_cell])

        logger.debug(prefix)
        logger.debug("Global Image Analysis Metrics:")
        logger.debug(data_global.iloc[0])

    if save_db != None:

        write_database(global_database, save_db)
        write_database(fibre_database, save_db, '_fibre')
        write_database(cell_database, save_db, '_cell')
