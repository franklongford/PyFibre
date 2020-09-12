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
from pyfibre.pyfibre_runner import PyFibreRunner

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
        self.supported_parsers = {}
        self.supported_readers = {}
        self.supported_analysers = {}

        for factory in factories:
            self.supported_parsers[factory.label] = factory.create_parser()
            self.supported_readers[factory.label] = factory.create_reader()
            self.supported_analysers[factory.label] = factory.create_analyser()

    def _run_pyfibre(self):

        input_files = []
        for file_path in self.file_paths:
            input_files += parse_file_path(file_path, self.key)

        file_sets = []
        for label, parser in self.supported_parsers.items():
            file_sets += parser.get_file_sets(input_files)

        for label, reader in self.supported_readers.items():

            logger.info(f"Analysing {label} images")
            analyser = self.supported_analysers[label]

            image_databases = [
                pd.DataFrame() for _ in analyser.database_names]

            generator = self.runner.run(
                file_sets, analyser, reader)

            for databases in generator:
                for index, database in enumerate(databases):
                    if isinstance(database, pd.Series):
                        image_databases[index] = image_databases[index].append(
                            database, ignore_index=True)
                    elif isinstance(database, pd.DataFrame):
                        image_databases[index] = pd.concat(
                            [image_databases[index], database])

            if self.database_name:
                for index, name in enumerate(analyser.database_names):
                    save_database(
                        image_databases[index],
                        self.database_name,
                        name)

    def run(self):

        if self.start():

            try:
                self._run_pyfibre()
            except Exception:
                logger.info('Error in PyFibre runner')
                raise

        self.stop()
