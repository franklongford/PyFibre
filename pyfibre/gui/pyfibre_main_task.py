import logging

import numpy as np
import pandas as pd

from pyface.api import ImageResource, FileDialog, OK
from pyface.qt import QtGui
from pyface.tasks.action.api import (
    SMenu, SMenuBar, SToolBar, TaskAction, TaskToggleGroup
)
from pyface.tasks.api import (
    PaneItem, Task, TaskLayout, Tabbed
)
from traits.api import (
    Bool, Int, List, Property, Instance, Event, Any,
    on_trait_change, HasStrictTraits, File
)
from traits_futures.api import (
    TraitsExecutor, CANCELLED, COMPLETED,
)

from pyfibre.gui.options_pane import OptionsPane
from pyfibre.gui.file_display_pane import FileDisplayPane
from pyfibre.gui.viewer_pane import ViewerPane
from pyfibre.io.database_io import save_database
from pyfibre.core.i_multi_image_factory import IMultiImageFactory
from pyfibre.core.pyfibre_runner import (
    PyFibreRunner)

logger = logging.getLogger(__name__)


def run_analysis(dictionary, runner, analysers, readers):

    for image_type, inner_dict in dictionary.items():
        analyser = analysers[image_type]
        reader = readers[image_type]
        list(runner.run(inner_dict, analyser, reader))


class PyFibreMainTask(Task):

    # ------------------
    # Regular Attributes
    # ------------------

    id = 'pyfibre.pyfibre_main_task'

    name = 'PyFibre GUI (Main)'

    database_filename = File('pyfibre_database')

    multi_image_factories = List(IMultiImageFactory)

    options_pane = Instance(OptionsPane)

    file_display_pane = Instance(FileDisplayPane)

    viewer_pane = Instance(ViewerPane)

    #: The Traits executor for the background jobs.
    traits_executor = Instance(TraitsExecutor, ())

    #: List of the submitted jobs, for display purposes.
    current_futures = List(Instance(HasStrictTraits))

    #: Maximum number of workers
    n_proc = Int(1)

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

        self.supported_readers = {}
        self.supported_analysers = {}

        for factory in self.multi_image_factories:
            self.supported_readers[factory.tag] = factory.create_reader()
            self.supported_analysers[factory.tag] = factory.create_analyser()

        self.image_databases = {
            tag: [None for _ in analyser.database_names]
            for tag, analyser in self.supported_analysers.items()
        }

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
        return FileDisplayPane(
            supported_readers=self.supported_readers)

    def _viewer_pane_default(self):
        return ViewerPane()

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
                        method="save_database_as"
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

    @on_trait_change('file_display_pane.selected_files')
    def update_selected_row(self):
        """Opens corresponding to the first item in
        selected_rows"""
        selected_row = (
            self.file_display_pane.selected_files[0])

        image_type = selected_row.tag
        file_names = selected_row.file_names

        try:
            reader = self.supported_readers[image_type]
            multi_image = reader.load_multi_image(
                file_names, selected_row.name)
        except (KeyError, ImportError):
            logger.debug(f'Cannot display image data for {file_names}')
        else:
            logger.info(f"Displaying image data for {file_names}")
            self.viewer_pane.update_viewer(
                multi_image, selected_row.name)

    @on_trait_change('run_enabled')
    def update_ui(self):
        if self.run_enabled:
            self.viewer_pane.update_image()

    @on_trait_change('current_futures:result_event')
    def _report_result(self, result):
        database = result[0]
        logger.info(f"Image analysis complete for {database['File']}")

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
            batch_dict = {tag: {} for tag in self.supported_readers.keys()}

            for row in batch_rows:
                batch_dict[row.tag].update(
                    {row.name: row.file_names})

            runner = PyFibreRunner(
                p_denoise=(self.options_pane.n_denoise,
                           self.options_pane.m_denoise),
                sigma=self.options_pane.sigma,
                alpha=self.options_pane.alpha,
                ow_network=self.options_pane.ow_network,
                ow_segment=self.options_pane.ow_segment,
                ow_metric=self.options_pane.ow_metric,
                save_figures=self.options_pane.save_figures)

            future = self.traits_executor.submit_iteration(
                run_analysis, batch_dict, runner,
                self.supported_analysers, self.supported_readers
            )
            self.current_futures.append(future)

    # ------------------
    #   Public Methods
    # ------------------

    def create_central_pane(self):
        """ Creates the central pane
        """
        return self.viewer_pane

    def create_dock_panes(self):
        """ Creates the dock panes
        """
        return [self.file_display_pane,
                self.options_pane]

    def create_databases(self):
        """Create and collate metric databases for each loaded multi
        image type"""

        image_databases = {
            tag: [pd.DataFrame() for _ in analyser.database_names]
            for tag, analyser in self.supported_analysers.items()
        }

        for row in self.file_display_pane.file_table:
            image_type = row.tag

            try:
                reader = self.supported_readers[image_type]
                analyser = self.supported_analysers[image_type]
            except KeyError:
                logger.info(
                    f"{row.name} analyser / reader not found"
                    f" - skipping")
                continue

            try:
                analyser.multi_image = reader.load_multi_image(
                    row.file_names, row.name)
                databases = analyser.load_databases()

                for index, database in enumerate(databases):
                    if isinstance(database, pd.Series):
                        image_databases[image_type][index] = (
                            image_databases[image_type][index].append(
                                database, ignore_index=True)
                        )
                    elif isinstance(database, pd.DataFrame):
                        image_databases[image_type][index] = (
                            pd.concat([image_databases[image_type][index],
                                       database], sort=True)
                        )

            except Exception:
                logger.info(
                    f"{row.name} databases not imported"
                    f" - skipping")

            finally:
                analyser.multi_image = None

        self.image_databases = image_databases

    def save_database(self, filename):
        """Save databases successfully generated by all loaded images"""

        for tag, analyser in self.supported_analysers.items():
            for index, name in enumerate(analyser.database_names):
                try:
                    save_database(
                        self.image_databases[tag][index],
                        filename,
                        name
                    )
                except IOError:
                    logger.exception("Error when saving databases")
                    return False
        return True

    def save_database_as(self):
        """ Shows a dialog to save the databases"""
        dialog = FileDialog(
            action="save as",
            default_filename=self.database_filename,
        )

        result = dialog.open()

        if result is not OK:
            return

        current_file = dialog.path

        self.create_databases()

        return self.save_database(current_file)

    def stop_run(self):
        self._cancel_all_fired()

    def exit_task(self):
        self.stop_run()
        self.traits_executor.stop()
