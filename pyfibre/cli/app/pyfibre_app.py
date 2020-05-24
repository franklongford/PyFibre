"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE

Created by: Frank Longford
Created on: 16/08/2018
"""
import logging

import pandas as pd

from envisage.api import Application
from traits.api import Instance, Str, List, File

from pyfibre.io.database_io import save_database
from pyfibre.io.utilities import parse_file_path
from pyfibre.ids import MULTI_IMAGE_FACTORIES
from .pyfibre_runner import PyFibreRunner, analysis_generator

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

        super(PyFibreApplication, self).__init__(
            runner=runner,
            **traits)

        factories = self.get_extensions(MULTI_IMAGE_FACTORIES)
        self.supported_readers = {}
        self.supported_analysers = {}

        for factory in factories:
            self.supported_readers[factory.tag] = factory.create_reader()
            self.supported_analysers[factory.tag] = factory.create_analyser()

    def _run_pyfibre(self):

        image_dictionary = {
            tag: {} for tag, _ in self.supported_readers.items()
        }

        for file_path in self.file_paths:
            input_files = parse_file_path(file_path, self.key)
            for tag, reader in self.supported_readers.items():
                image_dictionary[tag].update(
                    reader.collate_files(input_files)
                )

        for tag, inner_dict in image_dictionary.items():

            formatted_images = '\n'.join([
                f'\t{key}: {value}'
                for key, value in inner_dict.items()
            ])

            logger.info(f"Analysing {tag} images: \n{formatted_images}")

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
