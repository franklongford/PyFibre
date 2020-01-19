from pickle import UnpicklingError
from tkinter import Toplevel, Frame, TOP

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from pyfibre.io.database_io import check_file_name
from pyfibre.io.segment_io import load_segments
from pyfibre.model.objects.multi_image import MultiImage
from pyfibre.model.tools.analysis import tensor_analysis
from pyfibre.model.tools.metrics import fibre_metrics
from pyfibre.model.tools.filters import form_structure_tensor
from pyfibre.utilities import flatten_list


class PyFibreGraphs:

    def __init__(self, parent, width=750, height=750):

        self.parent = parent
        self.width = width
        self.height = height

        self.window = Toplevel(self.parent.master)
        self.window.tk.call('wm', 'iconphoto', self.window._w, self.parent.title.image)
        self.window.title('PyFibre - Graphs')
        self.window.geometry(f"{width}x{height}-100+40")

        self.frame = Frame(self.window)
        self._create_graphs()

    def _create_graphs(self):

        self.figure = Figure(figsize=(8, 4))
        self.angle_ax = self.figure.add_subplot(121, polar=True)
        self.angle_ax.set_title('Pixel Angle Histogram')

        self.fibre_ax = self.figure.add_subplot(122, polar=True)
        self.fibre_ax.set_title('Fibre Angle Histogram')

        self.fig_canvas = FigureCanvasTkAgg(self.figure, self.frame)
        self.fig_canvas.get_tk_widget().pack(side=TOP, fill="both", expand="yes")
        self.fig_canvas.draw()

        self.toolbar = NavigationToolbar2Tk(self.fig_canvas, self.frame)
        self.toolbar.update()
        self.toolbar.pack(side=TOP, fill="both", expand="yes")

        self.frame.pack()

    def display_figures(self):

        selected_file = self.parent.file_display.tree.selection()[0]

        image_name = selected_file.split('/')[-1]
        image_path = '/'.join(selected_file.split('/')[:-1])

        fig_name = check_file_name(image_name, extension='tif')
        data_dir = image_path + '/data/'

        file_index = self.parent.input_prefixes.index(selected_file)
        multi_image = MultiImage(
            self.parent.input_files[file_index],
            p_intensity=(self.parent.p0.get(), self.parent.p1.get()))

        if multi_image.shg_analysis:

            "Form nematic and structure tensors for each pixel"
            j_tensor = form_structure_tensor(multi_image.image_shg, sigma=1.0)

            "Perform anisotropy analysis on each pixel"
            pix_j_anis, pix_j_angle, pix_j_energy = tensor_analysis(j_tensor)

            pix_j_angle = (pix_j_angle.flatten() + 90) * np.pi / 180

            self.angle_ax.clear()
            self.angle_ax.set_title('Pixel Angle Histogram')
            self.angle_ax.hist(pix_j_angle, weights=pix_j_anis.flatten(), bins=50, density=True)
            # self.angle_ax.set_xlim(0, 180)

            try:
                fibres = load_segments(data_dir + fig_name + "_fibre")
                fibres = flatten_list(fibres)

                lengths, _, angles = fibre_metrics(fibres)

                self.fibre_ax.clear()
                self.fibre_ax.set_title('Fibre Angle Histogram')
                self.fibre_ax.hist(angles.flatten() * np.pi / 180, weights=lengths.flatten(),
                                   bins=50, density=True)
                # self.fibre_ax.set_xlim(0, 180)

                print("Displaying fibres for {}".format(fig_name))
            except (UnpicklingError, IOError, EOFError):
                self.fibre_tab.canvas.delete('all')
                print("Unable to display fibres for {}".format(fig_name))

        self.fig_canvas.draw()
