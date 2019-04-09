import os, sys, time
from tkinter import *
from tkinter import ttk, filedialog

import queue, threading
from multiprocessing import Pool, Process, JoinableQueue, Queue, current_process

import matplotlib
matplotlib.use("Agg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from PIL import ImageTk, Image
import networkx as nx
import numpy as np
import pandas as pd
from pickle import UnpicklingError

from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation

from skimage import img_as_float, measure
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import threshold_otsu
from skimage.color import gray2rgb, label2rgb
from skimage.restoration import (estimate_sigma, denoise_tv_chambolle, denoise_bilateral)

from main import analyse_image
import utilities as ut
from preprocessing import load_shg_pl, clip_intensities
from segmentation import draw_network
from figures import create_tensor_image, create_region_image, create_network_image


class pyfibre_gui:

	def __init__(self, master, n_proc, n_thread):

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
		self.n_thread = n_thread

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
		self.toggle.configure(background='#d8baa9')
		self.toggle.display_button = Button(self.toggle, width=15,
				   text="Viewer",
				   command=self.create_image_display)
		self.toggle.display_button.pack()
		self.toggle.place(x=300, y=10, height=50, width=250)

		self.file_display = Frame(self.master)
		self.create_file_display(self.file_display)
		self.file_display.place(x=5, y=220, height=600, width=800)


	def create_title(self, frame):

		self.master.title("PyFibre - Python Fibrous Image Analysis Toolkit")

		image = Image.open(self.pyfibre_dir + '/img/icon.ico')
		image = image.resize((300,200))
		image_tk = ImageTk.PhotoImage(image)

		self.master.tk.call('wm', 'iconphoto', self.master._w, image_tk)
		frame.text_title = Label(frame, image=image_tk)
		frame.image = image_tk
		frame.text_title.pack(side = TOP, fill = "both", expand = "yes")


	def create_options(self):

		try: self.options.window.lift()
		except (TclError, AttributeError): 
			self.options = pyfibre_options(self)


	def create_file_display(self, frame,  button_w= 18):

		frame.select_im_button = Button(frame, width=button_w,
				   text="Load Files",
				   command=self.add_images)
		frame.select_im_button.grid(column=0, row=0)

		frame.select_dir_button = Button(frame, width=button_w,
				   text="Load Folder",
				   command=self.add_directory)
		frame.select_dir_button.grid(column=1, row=0)

		frame.key = Entry(frame, width=button_w)
		frame.key.configure(background='#d8baa9')
		frame.key.grid(column=3, row=0, sticky=(N,W,E,S))

		frame.select_dir_button = Button(frame, width=button_w,
				   text="Filter",
				   command=lambda : self.del_images([filename for filename in self.input_prefixes \
							if (filename.find(frame.key.get()) == -1)]))
		frame.select_dir_button.grid(column=2, row=0)

		frame.delete_im_button = Button(frame, width=button_w,
				   text="Delete",
				   command=lambda : self.del_images(self.file_display.tree.selection()))
		frame.delete_im_button.grid(column=4, row=0)

		frame.tree = ttk.Treeview(frame, columns=('shg', 'pl'))
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

		frame.progress = ttk.Progressbar(frame, orient=HORIZONTAL, length=400, mode='determinate')
		frame.progress.grid(column=0, row=3, columnspan=5)

		frame.configure(background='#d8baa9')


	def add_images(self):
		
		new_files = filedialog.askopenfilenames(filetypes = (("tif files","*.tif"), ("all files","*.*")))
		new_files = list(new_files)

		files, prefixes = ut.get_image_lists(new_files)

		self.add_files(files, prefixes)


	def add_directory(self):
		
		directory = filedialog.askdirectory()
		new_files = []
		for file_name in os.listdir(directory):
			if file_name.endswith('.tif'):
				if 'display' not in file_name: 
					new_files.append( directory + '/' + file_name)

		files, prefixes = ut.get_image_lists(new_files)

		self.add_files(files, prefixes)


	def add_files(self, files, prefixes):

		new_indices = [i for i, prefix in enumerate(prefixes)\
						 if prefix not in self.input_prefixes]
		new_files = [files[i] for i in new_indices]
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


	def create_image_display(self):

		try: self.viewer.window.lift()
		except (TclError, AttributeError):
			self.viewer = pyfibre_viewer(self)
			self.master.bind('<Double-1>', lambda e: self.viewer.display_notebook())
		

	def generate_db(self):

		global_database = pd.DataFrame()
		fibre_database = pd.DataFrame()
		cell_database = pd.DataFrame()

		for i, input_file_name in enumerate(self.input_prefixes):

			image_name = input_file_name.split('/')[-1]
			image_path = '/'.join(input_file_name.split('/')[:-1])
			data_dir = image_path + '/data/'
			metric_name = data_dir + ut.check_file_name(image_name, extension='tif')
			
			self.update_log("Loading metrics for {}".format(metric_name))

			try:
				data_global = pd.read_pickle('{}_global_metric.pkl'.format(metric_name))
				data_fibre = pd.read_pickle('{}_fibre_metric.pkl'.format(metric_name))
				data_cell = pd.read_pickle('{}_cell_metric.pkl'.format(metric_name))

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
		db_filename = ut.check_file_name(db_filename, extension='pkl')
		db_filename = ut.check_file_name(db_filename, extension='xls')

		self.global_database.to_pickle(db_filename + '.pkl')
		self.global_database.to_excel(db_filename + '.xls')

		self.fibre_database.to_pickle(db_filename + '_fibre.pkl')
		self.fibre_database.to_excel(db_filename + '_fibre.xls')

		self.cell_database.to_pickle(db_filename + '_cell.pkl')
		self.cell_database.to_excel(db_filename + '_cell.xls')
		

		self.update_log("Saving Database files {}".format(db_filename))


	def write_run(self):

		self.file_display.run_button.config(state=DISABLED)	
		self.file_display.stop_button.config(state=NORMAL)
		self.file_display.progress['maximum'] = len(self.input_files)

		#"""Multi Processor version
		proc_count = np.min((self.n_proc, len(self.input_files)))
		index_split = np.array_split(np.arange(len(self.input_prefixes)),
						proc_count)

		self.processes = []
		for indices in index_split:

			batch_files = [self.input_files[i] for i in indices]
			batch_prefixes = [self.input_prefixes[i] for i in indices]

			process = Process(target=image_analysis, 
					args=(batch_files, batch_prefixes,
					(self.p0.get(), self.p1.get()),
					(self.n.get(), self.m.get()),
					self.sigma.get(), self.alpha.get(),
					self.ow_metric.get(), self.ow_segment.get(),
					 self.ow_network.get(), self.ow_figure.get(), 
					self.queue, self.n_thread))
			process.daemon = True
			self.processes.append(process)

		for process in self.processes: process.start()
		#"""

		"""Serial Version
		self.process = Process(target=image_analysis, args=(self.input_files, self.ow_metric.get(),
														self.ow_network.get(), self.queue))
		self.process.daemon = True
		self.process.start()
		"""
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


def image_analysis(input_files, input_prefixes, p_intensity, p_denoise, sigma, alpha, 
			ow_metric, ow_segment, ow_network, ow_figure, 
			queue, threads):

	for input_file_names, prefix in zip(input_files, input_prefixes):

		image_path = '/'.join(prefix.split('/')[:-1])

		try:
			analyse_image(input_file_names, prefix, image_path,
					scale=1.25, p_intensity=p_intensity,
					p_denoise=p_denoise, sigma=sigma,
					alpha=alpha,
					ow_metric=ow_metric, ow_segment=ow_segment,
					ow_network=ow_network, ow_figure=ow_figure,
					threads=threads)
			queue.put("Analysis of {} complete".format(prefix))

		except Exception as err: queue.put("{} {}".format(err.message, prefix))


class pyfibre_options:

	def __init__(self, parent, width=320, height=620):

		self.parent = parent
		self.width = width
		self.height = height

		"Initialise option parameters"
		self.window = Toplevel(self.parent.master)
		self.window.tk.call('wm', 'iconphoto', self.window._w, self.parent.title.image)
		self.window.title('PyFibre - Options')
		self.window.geometry(f"{width}x{height}-100+40")
		self.window.configure(background='#d8baa9')

		self.frame = Frame(self.window)
		self.create_options()

	def create_options(self):

		self.title_sigma = Label(self.frame, text="Gaussian Std Dev (pix)")
		self.title_sigma.configure(background='#d8baa9')
		self.sigma = Scale(self.frame, from_=0, to=10, tickinterval=1, resolution=0.1, 
				length=300, orient=HORIZONTAL, variable=self.parent.sigma)

		self.title_sigma.grid(column=0, row=2, rowspan=1)
		self.sigma.grid(column=0, row=3, sticky=(N,W,E,S))

		self.title_p0 = Label(self.frame, text="Low Clip Intensity (%)")
		self.title_p0.configure(background='#d8baa9')
		self.p0 = Scale(self.frame, from_=0, to=100, tickinterval=10, 
				length=300, orient=HORIZONTAL, variable=self.parent.p0)

		self.title_p1 = Label(self.frame, text="High Clip Intensity (%)")
		self.title_p1.configure(background='#d8baa9')
		self.p1 = Scale(self.frame, from_=0, to=100,tickinterval=10, 
				length=300, orient=HORIZONTAL, variable=self.parent.p1)

		self.title_p0.grid(column=0, row=4, rowspan=1)
		self.p0.grid(column=0, row=5)
		self.title_p1.grid(column=0, row=6, rowspan=1)
		self.p1.grid(column=0, row=7)

		self.title_n = Label(self.frame, text="NL-Mean Neighbourhood 1 (pix)")
		self.title_n.configure(background='#d8baa9')
		self.n = Scale(self.frame, from_=0, to=100, tickinterval=10, 
				length=300, orient=HORIZONTAL, variable=self.parent.n)

		self.title_m = Label(self.frame, text="NL-Mean Neighbourhood 2 (pix)")
		self.title_m.configure(background='#d8baa9')
		self.m = Scale(self.frame, from_=0, to=100,tickinterval=10,
				length=300, orient=HORIZONTAL, variable=self.parent.m)

		self.title_n.grid(column=0, row=8, rowspan=1)
		self.n.grid(column=0, row=9)
		self.title_m.grid(column=0, row=10, rowspan=1)
		self.m.grid(column=0, row=11)

		self.title_alpha = Label(self.frame, text="Alpha network coefficient")
		self.title_alpha.configure(background='#d8baa9')
		self.alpha = Scale(self.frame, from_=0, to=1, tickinterval=0.1, resolution=0.01,
						length=300, orient=HORIZONTAL, variable=self.parent.alpha)

		self.title_alpha.grid(column=0, row=12, rowspan=1)
		self.alpha.grid(column=0, row=13)

		self.chk_metric = Checkbutton(self.frame, text="o/w metrics", variable=self.parent.ow_metric)
		self.chk_metric.configure(background='#d8baa9')
		self.chk_metric.grid(column=0, row=14, sticky=(N,W,E,S))

		self.chk_segment = Checkbutton(self.frame, text="o/w segment", variable=self.parent.ow_segment)
		self.chk_segment.configure(background='#d8baa9')
		self.chk_segment.grid(column=0, row=15, sticky=(N,W,E,S))

		self.chk_network = Checkbutton(self.frame, text="o/w network", variable=self.parent.ow_network)
		self.chk_network.configure(background='#d8baa9')
		self.chk_network.grid(column=0, row=16, sticky=(N,W,E,S))

		self.chk_figure = Checkbutton(self.frame, text="o/w figure", variable=self.parent.ow_figure)
		self.chk_figure.configure(background='#d8baa9')
		self.chk_figure.grid(column=0, row=17, sticky=(N,W,E,S))

		self.chk_db = Checkbutton(self.frame, text="Save Database", variable=self.parent.save_db)
		self.chk_db.configure(background='#d8baa9')
		self.chk_db.grid(column=0, row=18, sticky=(N,W,E,S))

		self.frame.configure(background='#d8baa9')
		self.frame.pack()


class pyfibre_viewer:

	def __init__(self, parent, width=750, height=750):

		self.parent = parent
		self.width = width
		self.height = height

		self.window = Toplevel(self.parent.master)
		self.window.tk.call('wm', 'iconphoto', self.window._w, self.parent.title.image)
		self.window.title('PyFibre - Viewer')
		self.window.geometry(f"{width}x{height}-100+40")

		self.notebook = ttk.Notebook(self.window)
		self.create_tabs()


	def create_tabs(self):

		self.shg_image_tab = ttk.Frame(self.notebook)
		self.pl_image_tab = ttk.Frame(self.notebook)
		self.tran_image_tab = ttk.Frame(self.notebook)
		self.tensor_tab = ttk.Frame(self.notebook)
		self.network_tab = ttk.Frame(self.notebook)
		self.segment_tab = ttk.Frame(self.notebook)
		self.fibre_tab = ttk.Frame(self.notebook)
		self.cell_tab = ttk.Frame(self.notebook)
		self.metric_tab = ttk.Frame(self.notebook)

		self.tab_dict = {'SHG Image' : self.shg_image_tab,
						 'PL Image'  : self.pl_image_tab,
						 'Transmission Image'  : self.tran_image_tab,
						 'Tensor Image': self.tensor_tab,
						 'Network' : self.network_tab,
						 'Fibre' :  self.fibre_tab,
						 'Fibre Segment' : self.segment_tab,
						 'Cell Segment' : self.cell_tab}
		
		for key, tab in self.tab_dict.items():
			self.notebook.add(tab, text=key)

			tab.canvas = Canvas(tab, width=self.width, height=self.height,
									scrollregion=(0,0,self.height + 50 ,self.width + 50))  
			tab.scrollbar = Scrollbar(tab, orient=VERTICAL, 
								command=tab.canvas.yview)
			tab.scrollbar.pack(side=RIGHT,fill=Y)
			tab.canvas['yscrollcommand'] = tab.scrollbar.set
			tab.canvas.pack(side = LEFT, fill = "both", expand = "yes")

			if key in ['Tensor Image']:

				tab.figure = Figure(figsize=(5, 5))
				tab.fig_canvas = FigureCanvasTkAgg(tab.figure, tab)  
				tab.fig_canvas.get_tk_widget().pack(side = RIGHT, fill = "both", expand = "yes")
				tab.fig_canvas.draw()

				#tab.fig_toolbar = NavigationToolbar2Tk(tab.fig_canvas, tab)
				#tab.fig_toolbar.update()
				#tab.fig_toolbar.pack(side = TOP, fill = "both", expand = "yes")

		
		self.notebook.add(self.metric_tab, text='Metrics')
		self.metric_tab.metric_dict = {
			'No. Fibres' : {"info" : "Number of extracted fibres", "metric" : IntVar(), "tag" : "network"},
			'SHG Angle SDI' : {"info" : "Angle spectrum SDI of total image", "metric" : DoubleVar(), "tag" : "texture"},
			'SHG Pixel Anisotropy' : {"info" : "Average anisotropy of all pixels in total image", "metric" : DoubleVar(), "tag" : "texture"},
			'SHG Anisotropy' : {"info" : "Anisotropy of total image", "metric" : DoubleVar(), "tag" : "texture"},
			'SHG Intensity Mean' : {"info" : "Average pixel intensity of total image", "metric" : DoubleVar(), "tag" : "texture"},
			'SHG Intensity STD' : {"info" : "Pixel intensity standard deviation of total image", "metric" : DoubleVar(), "tag" : "texture"},
			'SHG Intensity Entropy' : {"info" : "Average Shannon entropy of total image", "metric" : DoubleVar(), "tag" : "texture"},						
			'Fibre GLCM Contrast' : {"info" : "SHG GLCM angle-averaged contrast", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre GLCM Homogeneity' : {"info" : "SHG GLCM angle-averaged homogeneity", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre GLCM Dissimilarity' : {"info" : "SHG GLCM angle-averaged dissimilarity", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre GLCM Correlation' : {"info" : "SHG GLCM angle-averaged correlation", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre GLCM Energy' : {"info" : "SHG GLCM angle-averaged energy", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre GLCM IDM' : {"info" : "SHG GLCM angle-averaged inverse difference moment", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre GLCM Variance' : {"info" : "SHG GLCM angle-averaged variance", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre GLCM Cluster' : {"info" : "SHG GLCM angle-averaged clustering tendency", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre GLCM Entropy' : {"info" : "SHG GLCM angle-averaged entropy", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre Area' : {"info" : "Average number of pixels covered by fibres", "metric" : DoubleVar(), "tag" : "content"},			
			'Fibre Coverage' : {"info" : "Ratio of image covered by fibres", "metric" : DoubleVar(), "tag" : "content"},
			'Fibre Linearity' : {"info" : "Average fibre segment linearity", "metric" : DoubleVar(), "tag" : "shape"},
			'Fibre Eccentricity' : {"info" : "Average fibre segment eccentricity", "metric" : DoubleVar(), "tag" : "shape"},
			'Fibre Density' : {"info" : "Average image fibre density", "metric" : DoubleVar(), "tag" : "texture"},
			'Fibre Hu Moment 1'  : {"info" : "Average fibre segment Hu moment 1", "metric" : DoubleVar(), "tag" : "shape"},
			'Fibre Hu Moment 2'  : {"info" : "Average fibre segment Hu moment 2", "metric" : DoubleVar(), "tag" : "shape"},
			'Fibre Waviness' : {"info" : "Average fibre waviness", "metric" : DoubleVar(), "tag" : "content"},
			'Fibre Lengths' : {"info" : "Average fibre pixel length", "metric" : DoubleVar(), "tag" : "content"},
			'Fibre Cross-Link Density' : {"info" : "Average cross-links per fibre", "metric" : DoubleVar(), "tag" : "content"},
			'Network Degree' : {"info" : "Average fibre network number of edges per node", "metric" : DoubleVar(), "tag" : "network"},
			'Network Eigenvalue' : {"info" : "Max Eigenvalue of network", "metric" : DoubleVar(), "tag" : "network"},
			'Network Connectivity' : {"info" : "Average fibre network connectivity", "metric" : DoubleVar(), "tag" : "network"},

			'No. Cells' : {"info" : "Number of cell segments", "metric" : IntVar(), "tag" : "content"},
			'PL Angle SDI' : {"info" : "Angle spectrum SDI of total image", "metric" : DoubleVar(), "tag" : "texture"},
			'PL Pixel Anisotropy' : {"info" : "Average anisotropy of all pixels in total image", "metric" : DoubleVar(), "tag" : "texture"},
			'PL Anisotropy' : {"info" : "Anisotropy of total image", "metric" : DoubleVar(), "tag" : "texture"},
			'PL Intensity Mean' : {"info" : "Average pixel intensity of total image", "metric" : DoubleVar(), "tag" : "texture"},
			'PL Intensity STD' : {"info" : "Pixel intensity standard deviation of total image", "metric" : DoubleVar(), "tag" : "texture"},
			'PL Intensity Entropy' : {"info" : "Average Shannon entropy of total image", "metric" : DoubleVar(), "tag" : "texture"},						
			'Cell GLCM Contrast' : {"info" : "PL GLCM angle-averaged contrast", "metric" : DoubleVar(), "tag" : "texture"},
			'Cell GLCM Homogeneity' : {"info" : "PL GLCM angle-averaged homogeneity", "metric" : DoubleVar(), "tag" : "texture"},
			'Cell GLCM Dissimilarity' : {"info" : "PL GLCM angle-averaged dissimilarity", "metric" : DoubleVar(), "tag" : "texture"},
			'Cell GLCM Correlation' : {"info" : "PL GLCM angle-averaged correlation", "metric" : DoubleVar(), "tag" : "texture"},
			'Cell GLCM Energy' : {"info" : "PL GLCM angle-averaged energy", "metric" : DoubleVar(), "tag" : "texture"},
			'Cell GLCM IDM' : {"info" : "PL GLCM angle-averaged inverse difference moment", "metric" : DoubleVar(), "tag" : "texture"},
			'Cell GLCM Variance' : {"info" : "PL GLCM angle-averaged variance", "metric" : DoubleVar(), "tag" : "texture"},
			'Cell GLCM Cluster' : {"info" : "PL GLCM angle-averaged clustering tendency", "metric" : DoubleVar(), "tag" : "texture"},
			'Muscle GLCM Contrast' : {"info" : "PL GLCM angle-averaged contrast", "metric" : DoubleVar(), "tag" : "texture"},
			'Muscle GLCM Homogeneity' : {"info" : "PL GLCM angle-averaged homogeneity", "metric" : DoubleVar(), "tag" : "texture"},
			'Muscle GLCM Dissimilarity' : {"info" : "PL GLCM angle-averaged dissimilarity", "metric" : DoubleVar(), "tag" : "texture"},
			'Muscle GLCM Correlation' : {"info" : "PL GLCM angle-averaged correlation", "metric" : DoubleVar(), "tag" : "texture"},
			'Muscle GLCM Energy' : {"info" : "PL GLCM angle-averaged energy", "metric" : DoubleVar(), "tag" : "texture"},
			'Muscle GLCM IDM' : {"info" : "PL GLCM angle-averaged inverse difference moment", "metric" : DoubleVar(), "tag" : "texture"},
			'Muscle GLCM Variance' : {"info" : "PL GLCM angle-averaged variance", "metric" : DoubleVar(), "tag" : "texture"},
			'Muscle GLCM Cluster' : {"info" : "PL GLCM angle-averaged clustering tendency", "metric" : DoubleVar(), "tag" : "texture"},
			'Cell Area' : {"info" : "Average number of pixels covered by cells", "metric" : DoubleVar(), "tag" : "content"},
			'Cell Linearity' : {"info" : "Average cell segment linearity", "metric" : DoubleVar(), "tag" : "shape"}, 
			'Cell Coverage' : {"info" : "Ratio of image covered by cell", "metric" : DoubleVar(), "tag" : "content"},		
			'Cell Eccentricity' : {"info" : "Average cell segment eccentricity", "metric" : DoubleVar(), "tag" : "shape"},				
			'Cell Density' : {"info" : "Average image cell density", "metric" : DoubleVar(), "tag" : "texture"},						
			'Cell Hu Moment 1'  : {"info" : "Average cell segment Hu moment 1", "metric" : DoubleVar(), "tag" : "shape"},
			'Cell Hu Moment 2'  : {"info" : "Average cell segment Hu moment 2", "metric" : DoubleVar(), "tag" : "shape"}
										}

		self.metric_tab.titles = list(self.metric_tab.metric_dict.keys())

		self.notebook.metrics = [DoubleVar() for i in range(len(self.metric_tab.titles))]
		self.metric_tab.headings = []
		self.metric_tab.info = []
		self.metric_tab.metrics = []

		self.metric_tab.texture = ttk.Labelframe(self.metric_tab, text="Texture",
						width=self.width-50, height=self.height-50)
		self.metric_tab.content = ttk.Labelframe(self.metric_tab, text="Content",
						width=self.width-50, height=self.height-50)
		self.metric_tab.shape = ttk.Labelframe(self.metric_tab, text="Shape",
						width=self.width-50, height=self.height-50)
		self.metric_tab.network = ttk.Labelframe(self.metric_tab, text="Network",
						width=self.width-50, height=self.height-50)
		
		self.metric_tab.frame_dict = {"texture" : {'tab' : self.metric_tab.texture, "count" : 0},
									  "content" : {'tab' : self.metric_tab.content, "count" : 0},
									  "shape"  : {'tab' : self.metric_tab.shape, "count" : 0},
									  "network" : {'tab' : self.metric_tab.network, "count" : 0}}

		for i, metric in enumerate(self.metric_tab.titles):

			tag = self.metric_tab.metric_dict[metric]["tag"]

			self.metric_tab.headings += [Label(self.metric_tab.frame_dict[tag]['tab'], 
				text="{}:".format(metric), font=("Ariel", 8))]
			self.metric_tab.info += [Label(self.metric_tab.frame_dict[tag]['tab'], 
				text=self.metric_tab.metric_dict[metric]["info"], font=("Ariel", 8))]
			self.metric_tab.metrics += [Label(self.metric_tab.frame_dict[tag]['tab'], 
				textvariable=self.metric_tab.metric_dict[metric]["metric"], font=("Ariel", 8))]

			self.metric_tab.headings[i].grid(column=0, row=self.metric_tab.frame_dict[tag]['count'])
			self.metric_tab.info[i].grid(column=1, row=self.metric_tab.frame_dict[tag]['count'])
			self.metric_tab.metrics[i].grid(column=2, row=self.metric_tab.frame_dict[tag]['count'])
			self.metric_tab.frame_dict[tag]['count'] += 1

		self.metric_tab.texture.pack()
		self.metric_tab.content.pack()
		self.metric_tab.shape.pack()
		self.metric_tab.network.pack()
		
		self.log_tab = ttk.Frame(self.notebook)
		self.notebook.add(self.log_tab, text='Log')
		self.log_tab.text = Text(self.log_tab, width=self.width-25, height=self.height-25)
		self.log_tab.text.insert(END, self.parent.Log)
		self.log_tab.text.config(state=DISABLED)

		self.log_tab.scrollbar = Scrollbar(self.log_tab, orient=VERTICAL, 
							command=self.log_tab.text.yview)
		self.log_tab.scrollbar.pack(side=RIGHT,fill=Y)
		self.log_tab.text['yscrollcommand'] = self.log_tab.scrollbar.set

		self.log_tab.text.pack()

		self.notebook.pack()
		#frame.notebook.BFrame.configure(background='#d8baa9')


	def display_image(self, canvas, image, x=0, y=0):

		canvas.delete('all')

		canvas.create_image(x, y, image=image, anchor=NW)
		canvas.image = image
		canvas.pack(side = LEFT, fill = "both", expand = True)

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
		fig_name = ut.check_file_name(image_name, extension='tif')
		data_dir = image_path + '/data/'

		file_index = self.parent.input_prefixes.index(selected_file)
		image_shg, image_pl, image_tran = load_shg_pl(self.parent.input_files[file_index])

		shg_analysis = ~np.any(image_shg == None)
		pl_analysis = ~np.any(image_pl == None)
		
		if shg_analysis:
			self.image_shg = clip_intensities(image_shg, 
					p_intensity=(self.parent.p0.get(), self.parent.p1.get())) * 255.999
			image_pil = Image.fromarray(self.image_shg.astype('uint8'))
			image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
			shg_image_tk = ImageTk.PhotoImage(image_pil)
			
			self.display_image(self.shg_image_tab.canvas, shg_image_tk)
			self.update_log("Displaying SHG image {}".format(fig_name))

			self.display_tensor(self.tensor_tab.canvas, self.image_shg)
			self.update_log("Displaying SHG tensor image {}".format(fig_name))

			try:
				networks = ut.load_region(data_dir + fig_name + "_network")
				self.display_network(self.network_tab.canvas, self.image_shg, networks)
				self.update_log("Displaying network for {}".format(fig_name))
			except (UnpicklingError, IOError, EOFError):
				self.network_tab.canvas.delete('all')
				self.update_log("Unable to display network for {}".format(fig_name))

			try:
				fibres = ut.load_region(data_dir + fig_name + "_fibre")
				fibres = ut.flatten_list(fibres)
				self.display_network(self.fibre_tab.canvas, self.image_shg, fibres, 1)
				self.update_log("Displaying fibres for {}".format(fig_name))
			except (UnpicklingError, IOError, EOFError):
				self.fibre_tab.canvas.delete('all')
				self.update_log("Unable to display fibres for {}".format(fig_name))

			try:
				segments = ut.load_region(data_dir + fig_name + "_fibre_segment")
				self.display_regions(self.segment_tab.canvas, self.image_shg, segments)
				self.update_log("Displaying fibre segments for {}".format(fig_name))
			except (UnpicklingError, IOError, EOFError):
				self.segment_tab.canvas.delete('all')
				self.update_log("Unable to display fibre segments for {}".format(fig_name))

		if pl_analysis:
			self.image_pl = clip_intensities(image_pl, 
					p_intensity=(self.parent.p0.get(), self.parent.p1.get())) * 255.999
			image_pil = Image.fromarray(self.image_pl.astype('uint8'))
			image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
			pl_image_tk = ImageTk.PhotoImage(image_pil)
			self.display_image(self.pl_image_tab.canvas, pl_image_tk)
			self.update_log("Displaying PL image {}".format(fig_name))

			self.image_tran = clip_intensities(image_tran, 
					p_intensity=(self.parent.p0.get(), self.parent.p1.get())) * 255.999
			image_pil = Image.fromarray(self.image_tran.astype('uint8'))
			image_pil = image_pil.resize((self.width, self.height), Image.ANTIALIAS)
			tran_image_tk = ImageTk.PhotoImage(image_pil)
			self.display_image(self.tran_image_tab.canvas, tran_image_tk)
			self.update_log("Displaying PL Transmission image {}".format(fig_name))
		
			try:	
				cells = ut.load_region(data_dir + fig_name + "_cell_segment")
				self.display_regions(self.cell_tab.canvas, self.image_pl, cells)
				self.update_log("Displaying cell segments for {}".format(fig_name))
			except (UnpicklingError, IOError, EOFError):
				self.cell_tab.canvas.delete('all')
				self.update_log("Unable to display cell segments for {}".format(fig_name))
		else: 
			self.pl_image_tab.canvas.delete('all')
			self.tran_image_tab.canvas.delete('all')
			self.cell_tab.canvas.delete('all')

		try:
			loaded_metrics = pd.read_pickle('{}_global_metric.pkl'.format(data_dir + fig_name)).iloc[0]
			for i, metric in enumerate(self.metric_tab.metric_dict.keys()):
				value = round(loaded_metrics[metric], 2)
				self.metric_tab.metric_dict[metric]["metric"].set(value)
			self.update_log("Displaying metrics for {}".format(fig_name))

		except (UnpicklingError, IOError, EOFError):
			self.update_log("Unable to display metrics for {}".format(fig_name))
			for i, metric in enumerate(self.metric_tab.titles):
				self.metric_tab.metric_dict[metric]["metric"].set(0)

		self.parent.master.update_idletasks()


N_PROC = 1#os.cpu_count() - 1
N_THREAD = 8

print(ut.logo())

root = Tk()
GUI = pyfibre_gui(root, N_PROC, N_THREAD)

root.mainloop()
