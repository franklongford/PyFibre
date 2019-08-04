import os
from functools import partial

from pyface.tasks.api import TraitsDockPane
from pyface.api import ImageResource

from traits.api import (
    HasTraits, List, Unicode, Button, File, Dict,
    Instance, Bool, on_trait_change, Int
)
from traitsui.api import (
    View, Item, ListEditor, Group, TableEditor, ObjectColumn,
    FileEditor, VGroup, HGroup, Spring, UItem, ImageEditor,
    TextEditor, ProgressEditor
)

from pyfibre.io.tif_reader import TIFReader
from pyfibre.io.utils import parse_files, parse_file_path


def horizontal_centre(item_or_group):
    return HGroup(Spring(), item_or_group, Spring())


class TableRow(HasTraits):

    name = Unicode()

    shg = Bool(False)

    pl = Bool(False)


class FileDisplayPane(TraitsDockPane):

    id = 'pyfibre.file_display_pane'

    name = 'File Display Pane'

    input_files = List(Unicode)

    input_prefixes = List(Unicode)

    file_table = List(TableRow)

    selected_files = List(TableRow)

    file_search = File()

    tif_reader = TIFReader(pl=True, shg=True)

    add_file_button = Button(name='Add Files')

    remove_file_button = Button(name='Remove Files')

    key = Unicode()

    progress = Int(0)

    #: The PyFibre logo. Stored at images/icon.ico
    image = ImageResource('icon.ico')

    file_list = Dict()

    def __init__(self, *args, **kwargs):
        super(FileDisplayPane, self).__init__(*args, **kwargs)

    def default_traits_view(self):

        table_editor = TableEditor(
            columns=[
                ObjectColumn(name="name",
                             label="name",
                             resize_mode="stretch"),
                ObjectColumn(name="shg",
                             label="shg",
                             resize_mode="stretch"),
                ObjectColumn(name="pl",
                             label="pl",
                             resize_mode="stretch")
            ],
            auto_size=False,
            selected='selected_files',
            on_select=self.open_file,
            selection_mode='rows',
            editable=False
        )

        file_editor = FileEditor(
            allow_dir=True,
            filter=['*.tif']
        )

        image_editor = ImageEditor(scale=True,
                                   allow_upscaling=False,
                                   preserve_aspect_ratio=True)

        progress_editor = ProgressEditor(min=0, max=100)

        traits_view = View(
            VGroup(
                Group(
                    UItem('image',
                          editor=image_editor
                          )
                ),
                Group(
                    Item('key',
                         editor=TextEditor(),
                         style='simple'),
                ),
                Group(
                    Item('file_search',
                         editor=file_editor,
                         style='custom'),
                    Item('add_file_button',
                         label='Add File'),
                    show_labels=False
                ),
                Group(
                    Item('file_table',
                         editor=table_editor),
                    Item('remove_file_button',
                         label='Remove File'),
                    show_labels=False
                ),
                Group(
                    Item('progress',
                         editor=progress_editor
                         ),
                    show_labels=False,
                )
            ),
            style='custom',
            resizable=True
        )

        return traits_view

    def _add_file_button_fired(self):

        self.add_files(self.file_search)

    def _remove_file_button_fired(self):

        self.remove_file(self.selected_files)

    def add_files(self, file_path):

        file_name, directory = parse_file_path(file_path)
        input_files = parse_files(file_name, directory, self.key)

        input_files = [file for file in input_files
                       if file not in self.input_files]

        self.tif_reader.get_image_lists(input_files)

        input_prefixes = [
            prefix for prefix, _ in self.tif_reader.files.items()
        ]

        self.input_files += input_files
        self.input_prefixes += input_prefixes

        print(self.input_files)

        self.file_table = []
        for key, data in self.tif_reader.files.items():
            keys = data.keys()
            shg = 'PL-SHG' in keys or 'SHG' in keys
            pl = 'PL-SHG' in keys or 'PL' in keys

            self.file_table.append(
                TableRow(
                    name=key,
                    shg=shg,
                    pl=pl
                )
            )

        self.tif_reader.load_multi_images()

    def open_file(self, selected_rows):
        """Opens corresponding to the first item in
        selected_rows"""

        prefix = selected_rows[0].name
        multi_image = self.tif_reader.files[prefix]['image']
        self.task.window.central_pane.selected_image = multi_image

    def remove_file(self, selected_rows):

        for selected_row in selected_rows:
            self.file_table.remove(selected_row)
            prefix = selected_row.name
            self.tif_reader.files.pop(prefix, None)