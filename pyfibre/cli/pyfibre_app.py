"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE

Created by: Frank Longford
Created on: 16/08/2018
"""
import logging

import pandas as pd

from envisage.api import Application
from envisage.core_plugin import CorePlugin
from traits.api import Instance, Str, List, File

from pyfibre.shg_pl_trans.shg_pl_trans_factory import (
    SHGPLTransFactory)
from pyfibre.io.database_io import save_database
from pyfibre.shg_pl_trans.shg_pl_reader import (
    collate_image_dictionary
)
from pyfibre.io.utilities import parse_file_path
from pyfibre.pyfibre_runner import PyFibreRunner, analysis_generator

logger = logging.getLogger(__name__)


class PyFibreApplication(Application):

    id = 'pyfibre.application'

    name = 'PyFibre CLI'

    runner = Instance(PyFibreRunner)

    key = Str

    database_name = Str

    file_paths = List(File)

    def __init__(self, sigma=0.5, alpha=0.5,
                 ow_metric=False, ow_segment=False,
                 ow_network=False, save_figures=False,
                 **traits):

        runner = PyFibreRunner(
            sigma=sigma, alpha=alpha,
            ow_metric=ow_metric, ow_segment=ow_segment,
            ow_network=ow_network, save_figures=save_figures
        )

        plugins = [CorePlugin()]

        super(PyFibreApplication, self).__init__(
            runner=runner,
            plugins=plugins,
            **traits)

        factory = SHGPLTransFactory()

        self.supported_readers = factory.create_reader()
        self.supported_analysers = factory.create_analyser()

    def _run_pyfibre(self):

        image_dictionary = {}

        for file_path in self.file_paths:
            input_files = parse_file_path(file_path, self.key)
            image_dictionary.update(collate_image_dictionary(input_files))

        formatted_images = '\n'.join([
            f'\t{key}: {value}'
            for key, value in image_dictionary.items()
        ])

        logger.info(f"Analysing images: \n{formatted_images}")

        global_database = pd.DataFrame()
        fibre_database = pd.DataFrame()
        network_database = pd.DataFrame()
        cell_database = pd.DataFrame()

        generator = analysis_generator(
            image_dictionary, self.runner,
            self.supported_analysers, self.supported_readers)

        for databases in generator:
            global_database = global_database.append(
                databases[0], ignore_index=True)

            fibre_database = pd.concat([fibre_database, databases[1]])
            network_database = pd.concat([network_database, databases[2]])
            cell_database = pd.concat([cell_database, databases[3]])

        if self.database_name:
            save_database(global_database, self.database_name)
            save_database(fibre_database, self.database_name, 'fibre')
            save_database(network_database, self.database_name, 'network')
            save_database(cell_database, self.database_name, 'cell')

    def run(self):

        if self.start():

            try:
                self._run_pyfibre()
            except Exception:
                logger.info('Error in PyFibre runner')
                raise

        self.stop()
