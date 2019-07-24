import logging
from multiprocessing import (
    Pool, Process, JoinableQueue, Queue, current_process
)

import numpy as np

from pyface.tasks.action.api import (
    SMenu, SMenuBar, SToolBar, TaskAction, TaskToggleGroup
)
from pyface.tasks.api import (
    PaneItem, Task, TaskLayout, VSplitter
)
from traits.api import (
    Bool, Int, List, Float, Instance
)

from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane
from pyfibre.gui.process_run import process_run

logger = logging.getLogger(__name__)


class PyFibreMainTask(Task):

    id = 'pyfibre.pyfibre_main_task'

    name = 'PyFibre GUI (Main)'

    options_pane = Instance(OptionsPane)

    file_display_pane = Instance(FileDisplayPane)

    # Multiprocessor list
    processes = Instance(List)

    #: The menu bar for this task.
    menu_bar = Instance(SMenuBar)

    tool_bars = List(SToolBar)

    def __init__(self, *args, **kwargs):
        super(PyFibreMainTask, self).__init__(*args, **kwargs)

        self.global_database = None
        self.fibre_database = None
        self.cell_database = None

        self.queue = Queue()
        self.processes = []

    def _default_layout_default(self):
        """ Defines the default layout of the task window """
        return TaskLayout(
            left=VSplitter(
                PaneItem('pyfibre.file_display_pane'),
                PaneItem('pyfibre.options_pane'))
        )

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
                        name="Save Database",
                        tooltip="Save database containing "
                                "image metrics",
                        method="save_database"
                    )
                )
            ]
        return tool_bars

    def create_central_pane(self):
        """ Creates the central pane
        """
        return ViewerPane()

    def create_dock_panes(self):
        """ Creates the dock panes
        """
        return [self.file_display_pane,
                self.options_pane]

    def _options_pane_default(self):
        return OptionsPane()

    def _file_display_pane_default(self):
        return FileDisplayPane()

    def _run_pyfibre(self):

        n_files = len(self.file_display_pane.input_files)
        proc_count = np.min(
            (self.n_proc, n_files)
        )
        index_split = np.array_split(np.arange(n_files), proc_count)

        self.processes = []

        for indices in index_split:
            batch_files = [self.file_display.input_files[i] for i in indices]

            process = Process(
                target=process_run,
                args=(
                    batch_files,
                    (self.options_pane.low_intensity,
                     self.options_pane.high_intensity),
                    (self.options_pane.n_denoise,
                     self.options_pane.m_denoise),
                    self.options_pane.sigma,
                    self.options_pane.alpha,
                    self.options_pane.ow_network,
                    self.options_pane.ow_metric,
                    self.options_pane.ow_segment,
                    self.options_pane.ow_figure,
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
        self.queue_check()

        if np.any([process.is_alive() for process in self.processes]):
            self.master.after(500, self._process_check)
        else:
            self.stop_run()
            self.generate_db()
            if self.save_db.get():
                self.save_database()

    def queue_check(self):

        while not self.queue.empty():
            try:
                msg = self.queue.get(0)
                self.update_log(msg)
                self.progress.configure(value=self.progress['value'] + 1)
                self.progress.update()
            except queue.Empty: pass

    def save_database(self):
        print('Saving database')
