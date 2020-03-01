"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 18/02/2019
"""

import logging
import click

from pyfibre.version import __version__
from pyfibre.tests.fixtures import test_image_path

from .pyfibre_cli import PyFibreCLI


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
            database_name, debug, profile,
            shg_analysis, pl_analysis, ow_metric, ow_segment,
            ow_network, save_figures, test):
    """Launches the PyFibre command line app"""

    run(file_path, key, sigma, alpha, log_name,
        database_name, debug, profile,
        shg_analysis, pl_analysis, ow_metric, ow_segment,
        ow_network, save_figures, test)


def run(file_path, key, sigma, alpha, log_name,
        database_name, debug, profile,
        shg_analysis, pl_analysis, ow_metric, ow_segment,
        ow_network, save_figures, test):

    if test:
        file_path = test_image_path
        debug = True
        profile = True
        shg_analysis = True
        pl_analysis = True
        ow_network = True
        save_figures = True

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(filename=f"{log_name}.log", filemode="w",
                        level=level)

    if profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()

    pyfibre_app = PyFibreCLI(
        sigma=sigma, alpha=alpha, key=key,
        database_name=database_name,
        shg_analysis=shg_analysis, pl_analysis=pl_analysis,
        ow_metric=ow_metric, ow_segment=ow_segment,
        ow_network=ow_network, save_figures=save_figures
    )

    pyfibre_app.run(file_path)

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
