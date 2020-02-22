import logging
import os
from queue import Empty
from multiprocessing import (
    Pool, Process, JoinableQueue, Queue, current_process
)

import numpy as np
import pandas as pd

from pyface.api import ImageResource
from pyface.tasks.action.api import (
    SMenu, SMenuBar, SToolBar, TaskAction, TaskToggleGroup
)
from pyface.tasks.api import (
    PaneItem, Task, TaskLayout, VSplitter, Tabbed
)
from pyface.timer.api import do_after
from traits.api import (
    Bool, Int, List, Float, Instance, Event, on_trait_change
)

from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane
from pyfibre.gui.process_run import process_run
from pyfibre.io.database_io import save_database, load_database
from pyfibre.model.image_analyser import ImageAnalyser
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

    # Multiprocessor list
    n_proc = Int(1)

    processes = List()

    progress_int = Int()

    #: The menu bar for this task.
    menu_bar = Instance(SMenuBar)

    tool_bars = List(SToolBar)

    run_enabled = Bool(True)

    change_options = Event()

    def __init__(self, *args, **kwargs):
        super(PyFibreMainTask, self).__init__(*args, **kwargs)

        self.global_database = None
        self.fibre_database = None
        self.cell_database = None

        self.queue = Queue()

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
                        name="Cancel",
                        tooltip="Cancel PyFibre",
                        image=ImageResource(
                            "baseline_pause_black_18dp"),
                        method="_stop_run",
                        image_size=(64, 64),
                        enabled_name='cancel_enabled'
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
    #   Private Methods
    # ------------------

    def _run_pyfibre(self):

        self.run_enabled = False

        if self.file_display_pane.n_images == 0:
            self._stop_run()
            return

        file_table = self.file_display_pane.file_table

        proc_count = np.min(
            (self.n_proc, self.file_display_pane.n_images))
        index_split = np.array_split(
            np.arange(self.file_display_pane.n_images),
            proc_count)

        self.processes = []

        for indices in index_split:
            batch_rows = [file_table[index] for index in indices]
            batch_dict = {row.name: row._dictionary
                          for row in batch_rows}

            image_analyser = ImageAnalyser(
                p_denoise=(self.options_pane.n_denoise,
                           self.options_pane.m_denoise),
                sigma=self.options_pane.sigma,
                alpha=self.options_pane.alpha,
                ow_network=self.options_pane.ow_network,
                ow_segment=self.options_pane.ow_segment,
                ow_metric=self.options_pane.ow_metric,
                save_figures=False)

            process = Process(
                target=process_run,
                args=(batch_dict,
                      image_analyser,
                      self.queue))
            process.daemon = True
            self.processes.append(process)

        for process in self.processes:
            process.start()

        self._process_check()

    def _process_check(self):
        """
        Check if there is something in the queue
        """

        self._queue_check()

        if np.any([process.is_alive() for process in self.processes]):
            do_after(1000, self._process_check)
        else:
            self._stop_run()
            self.create_databases()
            if self.options_pane.save_database:
                self.save_database()

    def _queue_check(self):

        while not self.queue.empty():
            msg = self.queue.get(0)
            logger.info(msg)

    def _stop_run(self):

        logger.info("Stopping Analysis")

        for process in self.processes:
            process.terminate()

        self.create_figures()
        self.file_display_pane.trait_set(progress=0)
        self.run_enabled = True

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
        image_analyser = ImageAnalyser(
            p_denoise=(self.options_pane.n_denoise,
                       self.options_pane.m_denoise),
            sigma=self.options_pane.sigma,
            alpha=self.options_pane.alpha)

        for row in file_table:

            reader.assign_images(row._dictionary)
            multi_image = reader.load_multi_image()

            (working_dir, data_dir, fig_dir,
             filename, figname) = image_analyser.get_filenames(row.name)

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

                global_database = global_database.append(data_global, ignore_index=True)
                fibre_database = pd.concat([fibre_database, data_fibre], sort=True)
                cell_database = pd.concat([cell_database, data_cell], sort=True)

            except (ValueError, IOError):
                logger.info(f"{input_file_name} databases not imported - skipping")

        self.global_database = global_database
        self.fibre_database = fibre_database
        self.cell_database = cell_database

    def save_database(self):

        filename = self.options_pane.database_filename

        save_database(self.global_database, filename)
        save_database(self.fibre_database, filename, 'fibre')
        save_database(self.cell_database, filename, 'cell')
