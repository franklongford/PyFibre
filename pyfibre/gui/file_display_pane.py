from pyface.tasks.api import TraitsDockPane
from pyface.api import ImageResource

from traits.api import (
    HasTraits, List, Unicode, Button, File, Dict,
    Int, Property, Str
)
from traitsui.api import (
    View, Item, Group, TableEditor, ObjectColumn,
    FileEditor, VGroup, HGroup, Spring, UItem, ImageEditor,
    TextEditor
)

from pyfibre.io.utilities import parse_file_path
from pyfibre.core.base_multi_image_reader import BaseMultiImageReader


def horizontal_centre(item_or_group):
    return HGroup(Spring(), item_or_group, Spring())


class TableRow(HasTraits):

    name = Unicode()

    tag = Str()

    file_names = List(File)


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

    supported_readers = Dict(Str, BaseMultiImageReader)

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
                ObjectColumn(name="tag",
                             label="File Type",
                             resize_mode="fixed")
            ],
            auto_size=False,
            selected='object.selected_files',
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
        """Adds a single file or directory to display

        Parameters
        ----------
        file_path: str
            Path to either a file or directory
        """

        input_prefixes = [row.name for row in self.file_table]

        input_files = parse_file_path(file_path)

        for tag, reader in self.supported_readers.items():
            image_dictionary = reader.collate_files(input_files)
            for prefix, file_names in image_dictionary.items():
                if prefix not in input_prefixes:
                    try:
                        reader.create_image_stack(file_names)
                        table_row = TableRow(
                            name=prefix,
                            tag=tag,
                            file_names=file_names)
                        self.file_table.append(table_row)
                    except Exception:
                        raise

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
