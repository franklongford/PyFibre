"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 18/02/2019
"""

import logging
import click

from pyfibre.core.base_pyfibre_plugin import BasePyFibrePlugin
from pyfibre.core.core_pyfibre_plugin import CorePyFibrePlugin
from pyfibre.utilities import logo, load_plugins

from ..version import __version__

from .pyfibre_cli import PyFibreApplication


@click.command()
@click.version_option(version=__version__)
@click.option(
    '--debug', is_flag=True, default=False,
    help="Prints extra debug information in pyfibre.log"
)
@click.option(
    '--profile', is_flag=True, default=False,
    help="Run GUI under cProfile, creating .prof and .pstats "
         "files in the current directory."
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
    default=''
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
    'file_paths', nargs=-1, type=click.Path(exists=True),
    required=False
)
def pyfibre(file_paths, key, sigma, alpha, log_name,
            database_name, debug, profile, ow_metric, ow_segment,
            ow_network, save_figures, test):
    """Launches the PyFibre command line app"""

    run(list(file_paths), key, sigma, alpha, log_name,
        database_name, debug, profile, ow_metric, ow_segment,
        ow_network, save_figures, test)


def run(file_paths, key, sigma, alpha, log_name,
        database_name, debug, profile,
        ow_metric, ow_segment,
        ow_network, save_figures, test):

    if test:
        debug = True
        profile = True
        ow_network = True
        save_figures = True

    if profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(filename=f"{log_name}.log",
                        filemode="w",
                        level=level)

    logging.info(logo(__version__))

    plugins = [CorePyFibrePlugin()]
    plugins += load_plugins()

    if test:
        file_paths = []
        for plugin in plugins:
            if isinstance(plugin, BasePyFibrePlugin):
                file_paths += plugin.get_test_files()

    pyfibre_app = PyFibreApplication(
        file_paths=file_paths,
        sigma=sigma, alpha=alpha, key=key,
        database_name=database_name,
        ow_metric=ow_metric, ow_segment=ow_segment,
        ow_network=ow_network, save_figures=save_figures,
        plugins=plugins
    )

    pyfibre_app.run()

    if profile:
        profiler.disable()
        from sys import version_info
        fname = 'pyfibre-{}-{}.{}.{}'.format(__version__,
                                             version_info.major,
                                             version_info.minor,
                                             version_info.micro)

        profiler.dump_stats(fname + '.prof')
        with open(fname + '.pstats', 'w') as fp:
            stats = pstats.Stats(
                profiler, stream=fp).sort_stats('cumulative')
            stats.print_stats()
