"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE 

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 18/02/2019
"""
import logging

import pandas as pd

from pyfibre.io.database_io import save_database
from pyfibre.io.shg_pl_reader import (
    collate_image_dictionary,
    SHGPLTransReader
)
from pyfibre.io.utilities import parse_files, parse_file_path
from pyfibre.model.image_analyser import ImageAnalyser
from pyfibre.model.iterator import iterate_images

logger = logging.getLogger(__name__)


class PyFibreCLI:

    id = 'pyfibre.pyfibre_cli'

    name = 'PyFibre CLI'

    def __init__(self, sigma=None, alpha=None, key=None,
                 database_name=None, shg_analysis=True,
                 pl_analysis=False, ow_metric=False,
                 ow_segment=False, ow_network=False,
                 save_figures=False):

        self.shg_analysis = shg_analysis
        self.pl_analysis = pl_analysis

        self.database_name = database_name
        self.key = key

        self.image_analyser = ImageAnalyser(
            sigma=sigma, alpha=alpha,
            shg_analysis=shg_analysis, pl_analysis=pl_analysis,
            ow_metric=ow_metric, ow_segment=ow_segment,
            ow_network=ow_network, save_figures=save_figures
        )
        self.reader = SHGPLTransReader()

    def run(self, file_path):

        file_name, directory = parse_file_path(file_path)
        input_files = parse_files(file_name, directory, self.key)

        image_dictionary = collate_image_dictionary(input_files)

        global_database = pd.DataFrame()
        fibre_database = pd.DataFrame()
        cell_database = pd.DataFrame()

        generator = iterate_images(
            image_dictionary, self.image_analyser, self.reader)

        for databases in generator:

            global_database = global_database.append(
                databases[0], ignore_index=True)

            if self.shg_analysis:
                fibre_database = pd.concat([fibre_database, databases[1]])
            if self.pl_analysis:
                cell_database = pd.concat([cell_database, databases[2]])

        if self.database_name:
            save_database(global_database, self.database_name)
            if self.shg_analysis:
                save_database(fibre_database, self.database_name, 'fibre')
            if self.pl_analysis:
                save_database(cell_database, self.database_name, 'cell')
