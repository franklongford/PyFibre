"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE

Created by: Frank Longford
Created on: 16/08/2018
"""
import logging

import pandas as pd

from pyfibre.io.database_io import save_database
from pyfibre.io.shg_pl_reader import (
    collate_image_dictionary,
    SHGPLTransReader
)
from pyfibre.io.utilities import parse_files, parse_file_path
from pyfibre.model.pyfibre_workflow import PyFibreWorkflow
from pyfibre.model.image_analyser import ImageAnalyser
from pyfibre.model.iterator import iterate_images

logger = logging.getLogger(__name__)


class PyFibreCLI:

    id = 'pyfibre.pyfibre_cli'

    name = 'PyFibre CLI'

    def __init__(self, sigma=0.5, alpha=0.5, key=None,
                 database_name=None, ow_metric=False,
                 ow_segment=False, ow_network=False,
                 save_figures=False):

        self.database_name = database_name
        self.key = key

        workflow = PyFibreWorkflow(
            sigma=sigma, alpha=alpha,
            ow_metric=ow_metric, ow_segment=ow_segment,
            ow_network=ow_network, save_figures=save_figures
        )
        self.image_analyser = ImageAnalyser(
            workflow=workflow
        )
        self.supported_readers = {
            'SHG-PL-Trans': SHGPLTransReader()
        }

    def run(self, file_path):

        file_name, directory = parse_file_path(file_path)
        input_files = parse_files(file_name, directory, self.key)
        image_dictionary = collate_image_dictionary(input_files)

        formatted_images = '\n'.join([
            f'\t{key}: {value}'
            for key, value in image_dictionary.items()
        ])

        logger.info(f"Analysing images: \n{formatted_images}")

        global_database = pd.DataFrame()
        fibre_database = pd.DataFrame()
        cell_database = pd.DataFrame()

        generator = iterate_images(
            image_dictionary, self.image_analyser, self.supported_readers)

        for databases in generator:

            global_database = global_database.append(
                databases[0], ignore_index=True)

            fibre_database = pd.concat([fibre_database, databases[1]])
            cell_database = pd.concat([cell_database, databases[2]])

        if self.database_name:
            save_database(global_database, self.database_name)
            save_database(fibre_database, self.database_name, 'fibre')
            save_database(cell_database, self.database_name, 'cell')
