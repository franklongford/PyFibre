from pyface.tasks.api import TraitsDockPane
from pyface.api import ImageResource

from traits.api import (
    HasTraits, List, Unicode, Button, File, Dict,
    Bool, Int, Property
)
from traitsui.api import (
    View, Item, Group, TableEditor, ObjectColumn,
    FileEditor, VGroup, HGroup, Spring, UItem, ImageEditor,
    TextEditor
)

from pyfibre.io.shg_pl_reader import (
    collate_image_dictionary)
from pyfibre.io.utilities import parse_files, parse_file_path


def horizontal_centre(item_or_group):
    return HGroup(Spring(), item_or_group, Spring())


class TableRow(HasTraits):

    name = Unicode()

    _dictionary = Dict()

    shg = Property(Bool, depends_on='_dictionary')

    pl = Property(Bool, depends_on='_dictionary')

    def _get_shg(self):
        return (
            'SHG-PL-Trans' in self._dictionary
            or 'SHG' in self._dictionary)

    def _get_pl(self):
        return (
            'SHG-PL-Trans' in self._dictionary
            or 'PL-Trans' in self._dictionary)


class FileDisplayPane(TraitsDockPane):

    # --------------------
    #  Regular Attributes
    # --------------------

    id = 'pyfibre.file_display_pane'

    name = 'File Display Pane'

    #: Remove the possibility to close the pane
    closable = False

    #: Remove the possibility to detach the pane from the GUI
    floatable = False

    #: Remove the possibility to move the pane in the GUI
    movable = False

    #: Make the pane visible by default
    visible = True

    file_table = List(TableRow)

    selected_files = List(TableRow)

    file_search = File()

    key = Unicode()

    #: The PyFibre logo. Stored at images/icon.ico
    image = ImageResource('icon.ico')

    n_images = Property(Int, depends_on='file_table[]')

    # --------------------
    #       Buttons
    # --------------------

    add_file_button = Button(name='Add Files')

    remove_file_button = Button(name='Remove Files')

    filter_file_button = Button(name='Filter Files')

    def default_traits_view(self):

        table_editor = TableEditor(
            columns=[
                ObjectColumn(name="name",
                             label="name",
                             resize_mode="stretch"),
                ObjectColumn(name="shg",
                             label="shg",
                             resize_mode="fixed"),
                ObjectColumn(name="pl",
                             label="pl",
                             resize_mode="fixed")
            ],
            auto_size=False,
            selected='selected_files',
            on_select=self.view_selected_row,
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

        traits_view = View(
            VGroup(
                Group(
                    UItem('image',
                          editor=image_editor
                          )
                ),
                HGroup(
                    Item('key',
                         editor=TextEditor(),
                         style='simple'),
                    Item('filter_file_button',
                         label='Filter'),
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
                )
            ),
            style='custom',
            resizable=True
        )

        return traits_view

    def _get_n_images(self):
        return len(self.file_table)

    def _add_file_button_fired(self):

        self.add_files(self.file_search)

    def _remove_file_button_fired(self):

        self.remove_file(self.selected_files)

    def _filter_file_button_fired(self):

        self.filter_files(self.key)

    def add_files(self, file_path):

        file_name, directory = parse_file_path(file_path)
        input_files = parse_files(file_name, directory, self.key)

        image_dictionary = collate_image_dictionary(input_files)

        input_prefixes = [row.name for row in self.file_table]

        for key, data in image_dictionary.items():
            if key not in input_prefixes:
                table_row = TableRow(
                    name=key,
                    _dictionary=data)
                if table_row.shg and table_row.pl:
                    self.file_table.append(table_row)

    def view_selected_row(self, selected_rows):
        """Opens corresponding to the first item in
        selected_rows"""
        self.task.window.central_pane.selected_row = selected_rows[0]

    def remove_file(self, selected_rows):
        for selected_row in selected_rows:
            self.file_table.remove(selected_row)

    def filter_files(self, key=None):

        if key == '':
            key = None

        if key is not None:
            selected_rows = []
            for row in self.file_table:
                prefix_name = row.name
                if (key not in prefix_name and
                        prefix_name not in selected_rows):
                    selected_rows.append(row)

            self.remove_file(selected_rows)
