import logging
import os

import numpy as np
import pandas as pd

from pyface.api import ImageResource
from pyface.qt import QtGui
from pyface.tasks.action.api import (
    SMenu, SMenuBar, SToolBar, TaskAction, TaskToggleGroup
)
from pyface.tasks.api import (
    PaneItem, Task, TaskLayout, Tabbed
)
from traits.api import (
    Bool, Int, List, Property, Instance, Event, Any,
    on_trait_change, HasStrictTraits
)
from traits_futures.api import (
    TraitsExecutor, CANCELLED, COMPLETED,
)

from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane
from pyfibre.io.database_io import save_database, load_database
from pyfibre.model.image_analyser import ImageAnalyser
from pyfibre.model.pyfibre_workflow import PyFibreWorkflow
from pyfibre.model.iterator import iterate_images
from pyfibre.io.shg_pl_reader import SHGPLTransReader


logger = logging.getLogger(__name__)


class PyFibreMainTask(Task):

    # ------------------
    # Regular Attributes
    # ------------------

    id = 'pyfibre.pyfibre_main_task'

    name = 'PyFibre GUI (Main)'

    options_pane = Instance(OptionsPane)

    file_display_pane = Instance(FileDisplayPane)

    #: The Traits executor for the background jobs.
    traits_executor = Instance(TraitsExecutor, ())

    #: List of the submitted jobs, for display purposes.
    current_futures = List(Instance(HasStrictTraits))

    #: Maximum number of workers
    n_proc = Int(2)

    #: The menu bar for this task.
    menu_bar = Instance(SMenuBar)

    #: The tool bar for this task.
    tool_bars = List(SToolBar)

    #: Is the run button enabled?
    run_enabled = Property(
        Bool(), depends_on='current_futures.state')

    #: Is the stop button enabled?
    stop_enabled = Property(
        Bool(), depends_on='current_futures.state')

    change_options = Event()

    # ------------------
    #   Private Traits
    # ------------------

    _progress_bar = Any()

    def __init__(self, *args, **kwargs):
        super(PyFibreMainTask, self).__init__(*args, **kwargs)

        self.global_database = None
        self.fibre_database = None
        self.cell_database = None

    # ------------------
    #     Defaults
    # ------------------

    def _default_layout_default(self):
        """ Defines the default layout of the task window """
        return TaskLayout(
            left=Tabbed(
                PaneItem('pyfibre.file_display_pane'),
                PaneItem('pyfibre.options_pane'))
        )

    def _options_pane_default(self):
        return OptionsPane()

    def _file_display_pane_default(self):
        return FileDisplayPane()

    def _menu_bar_default(self):
        """A menu bar with functions relevant to the Setup task.
        """
        menu_bar = SMenuBar(SMenu(TaskToggleGroup(),
                                  id='File', name='&File'),
                            SMenu(id='Edit', name='&Edit'),
                            SMenu(TaskToggleGroup(),
                                  id='View', name='&View'))

        return menu_bar

    def _tool_bars_default(self):
        tool_bars = [
                SToolBar(
                    TaskAction(
                        name="Run",
                        tooltip="Run PyFibre",
                        image=ImageResource(
                            "baseline_play_arrow_black_48dp"),
                        method="_run_pyfibre",
                        image_size=(64, 64),
                        enabled_name='run_enabled'
                    ),
                    TaskAction(
                        name="Stop",
                        tooltip="Stop PyFibre run",
                        image=ImageResource(
                            "baseline_stop_black_18dp"),
                        method="stop_run",
                        image_size=(64, 64),
                        enabled_name='stop_enabled'
                    ),
                    TaskAction(
                        name="Save Database",
                        tooltip="Save database containing "
                                "image metrics",
                        image=ImageResource(
                            "baseline_save_black_48dp"),
                        method="save_database"
                    )
                )
            ]
        return tool_bars

    # ------------------
    #  Listener Methods
    # ------------------

    def _get_run_enabled(self):
        if self.current_futures:
            return all([
                future.done
                for future in self.current_futures
            ])
        return True

    def _get_stop_enabled(self):
        if self.current_futures:
            return any([
                future.cancellable
                for future in self.current_futures
            ])
        return False

    @on_trait_change('current_futures:result_event')
    def _report_result(self, result):
        logger.info("Image analysis complete for {}".format(result))

    @on_trait_change('current_futures:done')
    def _future_done(self, future, name, new):
        if future.state == COMPLETED:
            print("Run complete")
            self.current_futures.remove(future)
        elif future.state == CANCELLED:
            print("Run cancelled")
            self.current_futures.remove(future)

    # ------------------
    #   Private Methods
    # ------------------

    def _create_progress_bar(self, dialog):
        self._progress_bar = QtGui.QProgressBar(dialog)
        return self._progress_bar

    def _cancel_all_fired(self):
        for future in self.current_futures:
            if future.cancellable:
                future.cancel()

    def _run_pyfibre(self):

        if self.file_display_pane.n_images == 0:
            self.stop_run()
            return

        file_table = self.file_display_pane.file_table

        proc_count = np.min(
            (self.n_proc, self.file_display_pane.n_images))
        index_split = np.array_split(
            np.arange(self.file_display_pane.n_images),
            proc_count)

        for indices in index_split:
            batch_rows = [file_table[index] for index in indices]
            batch_dict = {row.name: row._dictionary
                          for row in batch_rows}

            reader = SHGPLTransReader()
            workflow = PyFibreWorkflow(
                p_denoise=(self.options_pane.n_denoise,
                           self.options_pane.m_denoise),
                sigma=self.options_pane.sigma,
                alpha=self.options_pane.alpha,
                ow_network=self.options_pane.ow_network,
                ow_segment=self.options_pane.ow_segment,
                ow_metric=self.options_pane.ow_metric,
                save_figures=False)
            image_analyser = ImageAnalyser(
                workflow=workflow
            )

            future = self.traits_executor.submit_iteration(
                iterate_images, batch_dict, image_analyser, reader
            )
            self.current_futures.append(future)

    # ------------------
    #   Public Methods
    # ------------------

    def create_central_pane(self):
        """ Creates the central pane
        """
        return ViewerPane()

    def create_dock_panes(self):
        """ Creates the dock panes
        """
        return [self.file_display_pane,
                self.options_pane]

    def create_figures(self):

        file_table = self.file_display_pane.file_table
        reader = SHGPLTransReader()
        workflow = PyFibreWorkflow(
            p_denoise=(self.options_pane.n_denoise,
                       self.options_pane.m_denoise),
            sigma=self.options_pane.sigma,
            alpha=self.options_pane.alpha)
        image_analyser = ImageAnalyser(
            workflow=workflow
        )

        for row in file_table:

            reader.assign_images(row._dictionary)
            multi_image = reader.load_multi_image()

            filenames = image_analyser.get_filenames(row.name)
            (working_dir, data_dir, fig_dir,
             filename, figname) = filenames

            if not os.path.exists(fig_dir):
                os.mkdir(fig_dir)

            image_analyser.create_figures(
                multi_image, filename, figname)

    def create_databases(self):

        global_database = pd.DataFrame()
        fibre_database = pd.DataFrame()
        cell_database = pd.DataFrame()

        input_prefixes = [
            row.name for row in self.file_display_pane.file_table
        ]

        for i, input_file_name in enumerate(input_prefixes):

            image_name = os.path.basename(input_file_name)
            image_path = os.path.dirname(input_file_name)
            data_dir = image_path + '/data/'
            metric_name = data_dir + image_name

            logger.info("Loading metrics for {}".format(metric_name))

            try:
                data_global = load_database(metric_name, 'global_metric')
                data_fibre = load_database(metric_name, 'fibre_metric')
                data_cell = load_database(metric_name, 'cell_metric')

                global_database = global_database.append(
                    data_global, ignore_index=True)
                fibre_database = pd.concat(
                    [fibre_database, data_fibre], sort=True)
                cell_database = pd.concat(
                    [cell_database, data_cell], sort=True)

            except (ValueError, IOError):
                logger.info(
                    f"{input_file_name} databases not imported"
                    f" - skipping")

        self.global_database = global_database
        self.fibre_database = fibre_database
        self.cell_database = cell_database

    def save_database(self):

        filename = self.options_pane.database_filename

        save_database(self.global_database, filename)
        save_database(self.fibre_database, filename, 'fibre')
        save_database(self.cell_database, filename, 'cell')

    def stop_run(self):
        self._cancel_all_fired()

    def exit_task(self):
        self.stop_run()
        self.traits_executor.stop()
