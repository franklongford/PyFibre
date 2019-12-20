from PIL import ImageTk, Image
from tkinter import (
    Toplevel, Canvas, Scrollbar, Text, END, DISABLED, VERTICAL, RIGHT,
    LEFT, Y, NW, NORMAL
)
from tkinter.ttk import Notebook, Frame

from pickle import UnpicklingError

from pyfibre.io.segment_io import load_segment
from pyfibre.io.database_io import check_file_name
from pyfibre.model.tools.figures import (
    create_tensor_image, create_region_image, create_network_image
)
from pyfibre.io.multi_image import MultiImage
from pyfibre.utilities import flatten_list


class PyFibreViewer:

    def __init__(self, parent, width=750, height=750):

        self.parent = parent
        self.width = width
        self.height = height

        self.window = Toplevel(self.parent.master)
        self.window.tk.call('wm', 'iconphoto', self.window._w, self.parent.title.image)
        self.window.title('PyFibre - Viewer')
        self.window.geometry(f"{width}x{height}-100+40")

        self.notebook = Notebook(self.window)
        self.create_tabs()

    def create_tabs(self):

        self.shg_image_tab = Frame(self.notebook)
        self.pl_image_tab = Frame(self.notebook)
        self.tran_image_tab = Frame(self.notebook)
        self.tensor_tab = Frame(self.notebook)
        self.network_tab = Frame(self.notebook)
        self.segment_tab = Frame(self.notebook)
        self.fibre_tab = Frame(self.notebook)
        self.cell_tab = Frame(self.notebook)
        self.metric_tab = Frame(self.notebook)

        self.tab_dict = {'SHG Image': self.shg_image_tab,
                         'PL Image': self.pl_image_tab,
                         'Transmission Image': self.tran_image_tab,
                         'Tensor Image': self.tensor_tab,
                         'Network': self.network_tab,
                         'Fibre': self.fibre_tab,
                         'Fibre Segment': self.segment_tab,
                         'Cell Segment': self.cell_tab}

        for key, tab in self.tab_dict.items():
            self.notebook.add(tab, text=key)

            tab.canvas = Canvas(tab, width=self.width, height=self.height,
                                scrollregion=(0, 0, self.height + 50, self.width + 50))
            tab.scrollbar = Scrollbar(tab, orient=VERTICAL,
                                      command=tab.canvas.yview)
            tab.scrollbar.pack(side=RIGHT, fill=Y)
            tab.canvas['yscrollcommand'] = tab.scrollbar.set
            tab.canvas.pack(side=LEFT, fill="both", expand="yes")

        self.log_tab = Frame(self.notebook)
        self.notebook.add(self.log_tab, text='Log')
        self.log_tab.text = Text(self.log_tab, width=self.width - 25, height=self.height - 25)
        self.log_tab.text.insert(END, self.parent.Log)
        self.log_tab.text.config(state=DISABLED)

        self.log_tab.scrollbar = Scrollbar(self.log_tab, orient=VERTICAL,
                                           command=self.log_tab.text.yview)
        self.log_tab.scrollbar.pack(side=RIGHT, fill=Y)
        self.log_tab.text['yscrollcommand'] = self.log_tab.scrollbar.set

        self.log_tab.text.pack()

        self.notebook.pack()

    # frame.notebook.BFrame.configure(background='#d8baa9')

    def display_image(self, canvas, image, x=0, y=0):

        canvas.delete('all')

        canvas.create_image(x, y, image=image, anchor=NW)
        canvas.image = image
        canvas.pack(side=LEFT, fill="both", expand=True)

        self.parent.master.update_idletasks()

    def display_tensor(self, canvas, image):

        tensor_image = create_tensor_image(image) * 255.999

        image_pil = Image.fromarray(tensor_image.astype('uint8'))
        image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.display_image(canvas, image_tk)

    def display_network(self, canvas, image, networks, c_mode=0):

        image_network_overlay = create_network_image(image, networks, c_mode)

        image_pil = Image.fromarray(image_network_overlay.astype('uint8'))
        image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.display_image(canvas, image_tk)

    def display_regions(self, canvas, image, regions):

        image_label_overlay = create_region_image(image, regions) * 255.999

        image_pil = Image.fromarray(image_label_overlay.astype('uint8'))
        image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(image_pil)

        self.display_image(canvas, image_tk)

    def update_log(self, text):

        self.log_tab.text.config(state=NORMAL)
        self.parent.Log += text + '\n'
        self.log_tab.text.insert(END, text + '\n')
        self.log_tab.text.config(state=DISABLED)

    def display_notebook(self):

        selected_file = self.parent.file_display.tree.selection()[0]

        image_name = selected_file.split('/')[-1]
        image_path = '/'.join(selected_file.split('/')[:-1])
        fig_name = check_file_name(image_name, extension='tif')
        data_dir = image_path + '/data/'

        file_index = self.parent.input_prefixes.index(selected_file)
        self.multi_image = MultiImage(
            self.parent.input_files[file_index],
            p_intensity=(self.parent.p0.get(), self.parent.p1.get()))

        if self.multi_image.shg_analysis:
            image_shg = self.multi_image.image_shg * 255.999
            image_pil = Image.fromarray(image_shg.astype('uint8'))
            image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
            shg_image_tk = ImageTk.PhotoImage(image_pil)

            self.display_image(self.shg_image_tab.canvas, shg_image_tk)
            self.update_log("Displaying SHG image {}".format(fig_name))

            self.display_tensor(self.tensor_tab.canvas, image_shg)
            self.update_log("Displaying SHG tensor image {}".format(fig_name))

            try:
                networks = load_segment(data_dir + fig_name, "network")
                self.display_network(self.network_tab.canvas, image_shg, networks)
                self.update_log("Displaying network for {}".format(fig_name))
            except (UnpicklingError, IOError, EOFError):
                self.network_tab.canvas.delete('all')
                self.update_log("Unable to display network for {}".format(fig_name))

            try:
                fibres = load_segment(data_dir + fig_name, "fibre")
                fibres = flatten_list(fibres)
                self.display_network(self.fibre_tab.canvas, image_shg, fibres, 1)
                self.update_log("Displaying fibres for {}".format(fig_name))
            except (UnpicklingError, IOError, EOFError):
                self.fibre_tab.canvas.delete('all')
                self.update_log("Unable to display fibres for {}".format(fig_name))

            try:
                segments = load_segment(data_dir + fig_name, "fibre_segment")
                self.display_regions(self.segment_tab.canvas, image_shg, segments)
                self.update_log("Displaying fibre segments for {}".format(fig_name))
            except (AttributeError, UnpicklingError, IOError, EOFError):
                self.segment_tab.canvas.delete('all')
                self.update_log("Unable to display fibre segments for {}".format(fig_name))

        if self.multi_image.pl_analysis:
            image_pl = self.multi_image.image_pl * 255.999
            image_pil = Image.fromarray(image_pl.astype('uint8'))
            image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
            pl_image_tk = ImageTk.PhotoImage(image_pil)

            self.display_image(self.pl_image_tab.canvas, pl_image_tk)
            self.update_log("Displaying PL image {}".format(fig_name))

            image_tran = self.multi_image.image_tran * 255.999
            image_pil = Image.fromarray(image_tran.astype('uint8'))
            image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
            tran_image_tk = ImageTk.PhotoImage(image_pil)

            self.display_image(self.tran_image_tab.canvas, tran_image_tk)
            self.update_log("Displaying PL Transmission image {}".format(fig_name))

            try:
                cells = load_segment(data_dir + fig_name, "cell_segment")
                self.display_regions(self.cell_tab.canvas, image_pl, cells)
                self.update_log("Displaying cell segments for {}".format(fig_name))
            except (AttributeError, UnpicklingError, IOError, EOFError):
                self.cell_tab.canvas.delete('all')
                self.update_log("Unable to display cell segments for {}".format(fig_name))
        else:
            self.pl_image_tab.canvas.delete('all')
            self.tran_image_tab.canvas.delete('all')
            self.cell_tab.canvas.delete('all')

        self.parent.master.update_idletasks()

