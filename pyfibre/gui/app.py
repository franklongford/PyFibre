import os

from PIL import ImageTk, Image
import numpy as np
import pandas as pd

from tkinter import (
    BooleanVar, DoubleVar, IntVar, Frame, Label, Entry,
    TclError, TOP, Button, OUTSIDE, N, W, E, S, DISABLED,
    HORIZONTAL, NORMAL, Tk
)
from tkinter.ttk import Treeview, Progressbar
from tkinter import filedialog

import queue
from multiprocessing import Process, Queue

from pyfibre.cli.app import image_analysis
import pyfibre.utilities as ut
from .pyfibre_options import PyFibreOptions
from .pyfibre_metrics import PyFibreMetrics
from .pyfibre_graphs import PyFibreGraphs
from .pyfibre_viewer import PyFibreViewer
from pyfibre.io.database_io import (
    load_database, save_database, check_file_name
)
from pyfibre.io.tif_reader import TIFReader

import matplotlib
matplotlib.use("Agg")


class PyFibreGUI:

    def __init__(self, master, n_proc):
        """Main PyFibre GUI

        Parameters
        ----------
        master: Tk
            Master Tk object
        n_proc: int
            Number of processors to run parallel analysis on

        """

        "Set file locations"
        self.source_dir = os.path.dirname(os.path.realpath(__file__))
        self.pyfibre_dir = self.source_dir[:self.source_dir.rfind(os.path.sep)]
        self.current_dir = os.getcwd()

        "Initiatise program log, queue and input file list"
        self.Log = "Initiating PyFibre GUI\n"
        self.queue = Queue()
        self.input_files = []
        self.input_prefixes = []
        self.n_proc = n_proc

        "Initialise option variables"
        self.ow_metric = BooleanVar()
        self.ow_segment = BooleanVar()
        self.ow_network = BooleanVar()
        self.ow_figure = BooleanVar()
        self.save_db = BooleanVar()
        self.sigma = DoubleVar()
        self.sigma.set(0.5)
        self.p0 = IntVar()
        self.p0.set(1)
        self.p1 = IntVar()
        self.p1.set(99)
        self.n = IntVar()
        self.n.set(5)
        self.m = IntVar()
        self.m.set(35)
        self.alpha = DoubleVar()
        self.alpha.set(0.5)

        "Define GUI objects"
        self.master = master
        self.master.geometry("700x720")
        self.master.configure(background='#d8baa9')
        self.master.protocol("WM_DELETE_WINDOW", lambda: quit())

        self.title = Frame(self.master)
        self.create_title(self.title)
        self.title.place(bordermode=OUTSIDE, height=200, width=300)

        self.options = None
        self.toggle = Frame(self.master)
        self.toggle.configure(background='#d8baa9')

        self.toggle.options_button = Button(self.toggle, width=15,
                   text="Options",
                   command=self.create_options)
        self.toggle.options_button.pack()

        self.viewer = None
        self.toggle.viewer_button = Button(self.toggle, width=15,
                   text="Viewer",
                   command=self.create_image_viewer)
        self.toggle.viewer_button.pack()

        self.metrics = None
        self.toggle.metric_button = Button(self.toggle, width=15,
                   text="Metrics",
                   command=self.create_metric_display)
        self.toggle.metric_button.pack()

        self.graphs = None
        self.toggle.graph_button = Button(
            self.toggle, width=15,
            text="Graphs",
            command=self.create_graph_display)
        self.toggle.graph_button.pack()

        self.toggle.test_button = Button(self.toggle, width=15,
                   text="Test",
                   command=self.test_image)
        self.toggle.test_button.pack()

        self.toggle.place(x=300, y=10, height=140, width=250)

        self.file_display = Frame(self.master)
        self.create_file_display(self.file_display)
        self.file_display.place(x=5, y=220, height=600, width=800)

        self.master.bind('<Double-1>', lambda e: self.update_windows())

    def create_title(self, frame):

        self.master.title("PyFibre - Python Fibrous Image Analysis Toolkit")

        image = Image.open(self.pyfibre_dir + '/img/icon.ico')
        image = image.resize((300,200))
        image_tk = ImageTk.PhotoImage(image)

        self.master.tk.call('wm', 'iconphoto', self.master._w, image_tk)
        frame.text_title = Label(frame, image=image_tk)
        frame.image = image_tk
        frame.text_title.pack(side=TOP, fill="both", expand="yes")

    def create_options(self):

        try: self.options.window.lift()
        except (TclError, AttributeError):
            self.options = PyFibreOptions(self)

    def create_file_display(self, frame,  button_w= 18):

        frame.select_im_button = Button(
            frame, width=button_w,
            text="Load Files",
            command=self.add_images)
        frame.select_im_button.grid(column=0, row=0)

        frame.select_dir_button = Button(
            frame, width=button_w,
            text="Load Folder",
            command=self.add_directory)
        frame.select_dir_button.grid(column=1, row=0)

        frame.key = Entry(frame, width=button_w)
        frame.key.configure(background='#d8baa9')
        frame.key.grid(column=3, row=0, sticky=(N,W,E,S))

        frame.select_dir_button = Button(
            frame, width=button_w,
            text="Filter",
            command=lambda: self.del_images(
                [filename for filename in self.input_prefixes
                 if filename.find(frame.key.get()) == -1]
            )
        )
        frame.select_dir_button.grid(column=2, row=0)

        frame.delete_im_button = Button(frame, width=button_w,
                   text="Delete",
                   command=lambda : self.del_images(self.file_display.tree.selection()))
        frame.delete_im_button.grid(column=4, row=0)

        frame.tree = Treeview(frame, columns=('shg', 'pl'))
        frame.tree.column("#0", minwidth=20)
        frame.tree.column('shg', width=5, minwidth=5, anchor='center')
        frame.tree.heading('shg', text='SHG')
        frame.tree.column('pl', width=5, minwidth=5, anchor='center')
        frame.tree.heading('pl', text='PL')
        frame.tree.grid(column=0, row=1, columnspan=5, sticky=(N,W,E,S))

        frame.run_button = Button(frame, width=3*button_w,
                   text="GO",
                   command=self.write_run)
        frame.run_button.grid(column=0, row=2, columnspan=3)

        frame.stop_button = Button(frame, width=2*button_w,
                   text="STOP",
                   command=self.stop_run, state=DISABLED)
        frame.stop_button.grid(column=2, row=2, columnspan=2)

        frame.progress = Progressbar(frame, orient=HORIZONTAL, length=400, mode='determinate')
        frame.progress.grid(column=0, row=3, columnspan=5)

        frame.configure(background='#d8baa9')

    def add_images(self):

        new_files = filedialog.askopenfilenames(filetypes = (("tif files","*.tif"), ("all files","*.*")))
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
            self.file_display.tree.insert('', 'end', filename, text=filename)
            self.file_display.tree.set(filename, 'shg', 'X')
            if len(new_files[i]) == 1:
                if '-pl-shg' in new_files[i][0].lower():
                    self.file_display.tree.set(filename, 'pl', 'X')
                else:
                    self.file_display.tree.set(filename, 'pl', '')

            if len(new_files[i]) == 2:
                self.file_display.tree.set(filename, 'pl', 'X')

            self.update_log("Adding {}".format(filename))

    def del_images(self, file_list):

        for filename in file_list:
            index = self.input_prefixes.index(filename)
            self.input_files.remove(self.input_files[index])
            self.input_prefixes.remove(filename)
            self.file_display.tree.delete(filename)
            self.update_log("Removing {}".format(filename))

    def create_image_viewer(self):

        try: self.viewer.window.lift()
        except (TclError, AttributeError):
            self.viewer = PyFibreViewer(self)

    def create_metric_display(self):

        try: self.metrics.window.lift()
        except (TclError, AttributeError):
            self.metrics = PyFibreMetrics(self)

    def create_graph_display(self):

        try: self.graphs.window.lift()
        except (TclError, AttributeError):
            self.graphs = PyFibreGraphs(self)

    def update_windows(self):

        try: self.viewer.display_notebook()
        except (TclError, AttributeError): pass

        try: self.metrics.get_metrics()
        except (TclError, AttributeError): pass

        try: self.graphs.display_figures()
        except (TclError, AttributeError): pass

    def generate_db(self):

        global_database = pd.DataFrame()
        fibre_database = pd.DataFrame()
        cell_database = pd.DataFrame()

        for i, input_file_name in enumerate(self.input_prefixes):

            image_name = input_file_name.split('/')[-1]
            image_path = '/'.join(input_file_name.split('/')[:-1])
            data_dir = image_path + '/data/'
            metric_name = data_dir + check_file_name(image_name, extension='tif')

            self.update_log("Loading metrics for {}".format(metric_name))

            try:
                data_global = load_database(metric_name, '_global_metric')
                data_fibre = load_database(metric_name, '_fibre_metric')
                data_cell = load_database(metric_name, '_cell_metric')

                global_database = pd.concat([global_database, data_global], sort=True)
                fibre_database = pd.concat([fibre_database, data_fibre], sort=True)
                cell_database = pd.concat([cell_database, data_cell], sort=True)

            except (ValueError, IOError):
                self.update_log(f"{input_file_name} databases not imported - skipping")


        self.global_database = global_database
        self.fibre_database = fibre_database
        self.cell_database = cell_database

        #self.update_dashboard()

    def save_database(self):

        db_filename = filedialog.asksaveasfilename()

        save_database(self.global_database, db_filename)
        save_database(self.fibre_database, db_filename, '_fibre')
        save_database(self.cell_database, db_filename, '_cell')

        self.update_log("Saving Database files {}".format(db_filename))

    def write_run(self):

        self.file_display.run_button.config(state=DISABLED)
        self.file_display.stop_button.config(state=NORMAL)
        self.file_display.progress['maximum'] = len(self.input_files)

        proc_count = np.min((self.n_proc, len(self.input_files)))
        index_split = np.array_split(np.arange(len(self.input_files)),
                        proc_count)

        self.processes = []
        for indices in index_split:

            batch_files = [self.input_files[i] for i in indices]

            process = Process(target=image_analysis,
                    args=(batch_files,
                    (self.p0.get(), self.p1.get()),
                    (self.n.get(), self.m.get()),
                    self.sigma.get(), self.alpha.get(),
                    self.ow_metric.get(), self.ow_segment.get(),
                     self.ow_network.get(), self.ow_figure.get(),
                    self.queue))
            process.daemon = True
            self.processes.append(process)

        for process in self.processes: process.start()

        self.process_check()

    def process_check(self):
        """
        Check if there is something in the queue
        """
        self.queue_check()

        #if self.process.exitcode is None:
        if np.any([process.is_alive() for process in self.processes]):
            self.master.after(500, self.process_check)
        else:
            self.stop_run()
            self.generate_db()
            if self.save_db.get(): self.save_database()

    def queue_check(self):

        while not self.queue.empty():
            try:
                msg = self.queue.get(0)
                self.update_log(msg)
                self.file_display.progress.configure(value=self.file_display.progress['value'] + 1)
                self.file_display.progress.update()
            except queue.Empty: pass

    def stop_run(self):

        self.update_log("Stopping Analysis")
        for process in self.processes: process.terminate()
        self.file_display.progress['value'] = 0
        self.file_display.run_button.config(state=NORMAL)
        self.file_display.stop_button.config(state=DISABLED)


    def update_log(self, text):

        self.Log += text + '\n'

    def test_image(self):

        if self.file_display.run_button['state'] == NORMAL:
            input_files = [self.pyfibre_dir + '/tests/stubs/test-pyfibre-pl-shg-Stack.tif']

            self.add_files(input_files)

            self.file_display.run_button.config(state=DISABLED)
            self.file_display.stop_button.config(state=NORMAL)
            self.file_display.progress['maximum'] = len(input_files)

            proc_count = np.min((self.n_proc, len(input_files)))
            index_split = np.array_split(np.arange(len(input_files)), proc_count)

            self.processes = []
            for indices in index_split:

                batch_files = [input_files[i] for i in indices]

                process = Process(target=run_analysis,
                        args=(batch_files,
                        (self.p0.get(), self.p1.get()),
                        (self.n.get(), self.m.get()),
                        self.sigma.get(), self.alpha.get(),
                        self.ow_metric.get(), self.ow_segment.get(),
                         self.ow_network.get(), self.ow_figure.get(),
                        self.queue))
                process.daemon = True
                self.processes.append(process)

            for process in self.processes: process.start()

            self.process_check()


def run_analysis(input_files, p_intensity, p_denoise,
                 sigma, alpha, ow_metric, ow_segment, ow_network,
                 ow_figure, queue):

    reader = TIFReader(input_files, shg=True, pl=True, p_intensity=p_intensity,
                       ow_network=ow_network, ow_segment=ow_segment,
                       ow_metric=ow_metric, ow_figure=ow_figure)
    reader.load_multi_images()

    for prefix, data in reader.files.items():
        try:
            image_analysis(
                data['image'], prefix, scale=1.25,
                sigma=sigma, alpha=alpha, p_denoise=p_denoise
            )
            queue.put("Analysis of {} complete".format(prefix))

        except Exception as err:
            queue.put("Error occurred in analysis of {}".format(prefix))
            raise err


def run():

    N_PROC = 1#os.cpu_count() - 1

    print(ut.logo())

    root = Tk()
    GUI = PyFibreGUI(root, N_PROC)

    root.mainloop()
