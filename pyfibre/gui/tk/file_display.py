import os
import logging
from tkinter import Frame, Button, Entry, N, W, E, S
from tkinter.ttk import Treeview
from tkinter import filedialog

from pyfibre.io.tif_reader import TIFReader

logger = logging.getLogger(__name__)


class FileDisplay(Frame):

    def _init__(self, master):
        super(FileDisplay, self).__init__(master)

        button_w = 18
        self.input_files = []
        self.input_prefixes = []

        self.select_im_button = Button(
            self, width=button_w,
            text="Load Files",
            command=self.add_images)
        self.select_im_button.grid(column=0, row=0)

        self.select_dir_button = Button(
            self, width=button_w,
            text="Load Folder",
            command=self.add_directory)
        self.select_dir_button.grid(column=1, row=0)

        self.key = Entry(self, width=button_w)
        self.key.configure(background='#d8baa9')
        self.key.grid(column=3, row=0, sticky=(N, W, E, S))

        self.select_dir_button = Button(
            self, width=button_w,
            text="Filter",
            command=lambda: self.del_images(
                [filename for filename in self.input_prefixes
                 if filename.find(self.key.get()) == -1]
            )
        )
        self.select_dir_button.grid(column=2, row=0)

        self.tree = Treeview(self, columns=('shg', 'pl'))

        self.delete_im_button = Button(
            self, width=button_w, text="Delete",
            command=lambda: self.del_images(
                self.tree.selection()
            )
        )
        self.delete_im_button.grid(column=4, row=0)

        self.tree.column("#0", minwidth=20)
        self.tree.column('shg', width=5, minwidth=5, anchor='center')
        self.tree.heading('shg', text='SHG')
        self.tree.column('pl', width=5, minwidth=5, anchor='center')
        self.tree.heading('pl', text='PL')
        self.tree.grid(column=0, row=1, columnspan=5, sticky=(N,W,E,S))

        self.configure(background='#d8baa9')

    def add_images(self):

        new_files = filedialog.askopenfilenames(
            filetypes = (("tif files","*.tif"), ("all files","*.*")))
        new_files = list(new_files)

        self.add_files(new_files)

    def add_directory(self):

        directory = filedialog.askdirectory()
        new_files = []
        for file_name in os.listdir(directory):
            if file_name.endswith('.tif'):
                if 'display' not in file_name:
                    new_files.append( directory + '/' + file_name)

        self.add_files(new_files)

    def add_files(self, input_files):

        reader = TIFReader(input_files, shg=True, pl=True)
        prefixes = [prefix for prefix, _ in reader.files.items()]

        new_indices = [i for i, prefix in enumerate(prefixes)\
                         if prefix not in self.input_prefixes]
        new_files = [input_files[i] for i in new_indices]
        new_prefixes = [prefixes[i] for i in new_indices]

        self.input_files += new_files
        self.input_prefixes += new_prefixes

        for i, filename in enumerate(new_prefixes):
            self.tree.insert('', 'end', filename, text=filename)
            self.tree.set(filename, 'shg', 'X')
            if len(new_files[i]) == 1:
                if '-pl-shg' in new_files[i][0].lower():
                    self.tree.set(filename, 'pl', 'X')
                else:
                    self.tree.set(filename, 'pl', '')

            if len(new_files[i]) == 2:
                self.tree.set(filename, 'pl', 'X')

            logger.info("Adding {}".format(filename))

    def del_images(self, file_list):

        for filename in file_list:
            index = self.input_prefixes.index(filename)
            self.input_files.remove(self.input_files[index])
            self.input_prefixes.remove(filename)
            self.tree.delete(filename)
            logger.info("Removing {}".format(filename))