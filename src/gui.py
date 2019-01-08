import matplotlib
matplotlib.use("Agg")

import os, sys, time
from tkinter import *
from tkinter import ttk, filedialog
import queue, threading
from multiprocessing import Pool, Process, JoinableQueue, Queue, current_process

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PIL import ImageTk, Image
import networkx as nx
import numpy as np
import pandas as pd

from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter

from skimage import img_as_float
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from skimage.color import gray2rgb
from skimage.restoration import (estimate_sigma, denoise_tv_chambolle, denoise_bilateral)

from main import *
import utilities as ut
from analysis import analyse_image, NoiseError

class imagecol_gui:

	def __init__(self, master, n_proc):

		"Set file locations"
		self.dir_path = os.path.dirname(os.path.realpath(__file__))
		self.current_dir = os.getcwd()
		"Initiatise program log, queue and input file list"
		self.Log = "Initiating PyFibre GUI\n"
		self.queue = Queue()
		self.input_files = []
		self.n_proc = n_proc

		"Define GUI objects"
		self.master = master
		self.master.geometry("1055x620")
		self.master.configure(background='#d8baa9')
		self.master.protocol("WM_DELETE_WINDOW", lambda: quit())

		self.title = Frame(self.master)
		self.create_title(self.title)
		self.title.place(bordermode=OUTSIDE, height=200, width=300)

		self.options = Frame(self.master)
		self.create_options(self.options)
		self.options.place(x=300, y=1, height=200, width=150)

		self.file_display = Frame(self.master)
		self.create_file_display(self.file_display)
		self.file_display.place(x=5, y=220, height=600, width=445)

		self.image_display = ttk.Notebook(self.master)
		self.create_image_display(self.image_display)
		self.master.bind('<Double-1>', lambda e: self.display_notebook())
		self.image_display.place(x=450, y=10, width=625, height=600)


	def create_title(self, frame):

		self.master.title("PyFibre - Python Fibrous Image Analysis Toolkit")

		image = Image.open(self.dir_path + '/icon.ico')
		image = image.resize((300,200))
		image_tk = ImageTk.PhotoImage(image)

		self.master.tk.call('wm', 'iconphoto', self.master._w, image_tk)
		frame.text_title = Label(frame, image=image_tk)
		frame.image = image_tk
		frame.text_title.pack(side = TOP, fill = "both", expand = "yes")


	def create_options(self, frame):

		frame.title = Label(frame, text="Options")
		frame.title.configure(background='#d8baa9')
		frame.title.grid(column=0, row=0, rowspan=2)

		self.ow_metric = IntVar()
		frame.chk_anis = Checkbutton(frame, text="o/w metrics", variable=self.ow_metric)
		frame.chk_anis.configure(background='#d8baa9')
		frame.chk_anis.grid(column=0, row=2, sticky=(N,W,E,S))
		#frame.chk_anis.pack(side=LEFT)

		self.ow_network = IntVar()
		frame.chk_graph = Checkbutton(frame, text="o/w graph", variable=self.ow_network)
		frame.chk_graph.configure(background='#d8baa9')
		frame.chk_graph.grid(column=0, row=3, sticky=(N,W,E,S))
		#frame.chk_graph.pack(side=LEFT)

		self.save_db = IntVar()
		frame.chk_db = Checkbutton(frame, text="Save Database", variable=self.save_db)
		frame.chk_db.configure(background='#d8baa9')
		frame.chk_db.grid(column=0, row=4, sticky=(N,W,E,S))

		frame.configure(background='#d8baa9')


	def create_file_display(self, frame):

		frame.select_im_button = Button(frame, width=15,
				   text="Select files",
				   command=self.add_images)
		frame.select_im_button.grid(column=0, row=0)

		frame.select_dir_button = Button(frame, width=15,
				   text="Select directory",
				   command=self.add_directory)
		frame.select_dir_button.grid(column=1, row=0)

		frame.delete_im_button = Button(frame, width=15,
				   text="Delete",
				   command=self.del_images)
		frame.delete_im_button.grid(column=2, row=0)

		frame.file_box = Listbox(frame, height=20, width=40, selectmode="extended")
		frame.file_box.grid(column=0, row=1, columnspan=3, sticky=(N,W,E,S))

		frame.scrollbar = ttk.Scrollbar(frame, orient=VERTICAL, command=frame.file_box.yview)
		frame.scrollbar.grid(column=3, row=1, sticky=(N,S))
		frame.file_box['yscrollcommand'] = frame.scrollbar.set

		#frame.grid_columnconfigure(1, weight=1)
		#frame.grid_rowconfigure(1, weight=1)

		frame.run_button = Button(frame, width=30,
				   text="GO",
				   command=self.write_run)
		frame.run_button.grid(column=0, row=2, columnspan=2)

		frame.stop_button = Button(frame, width=15,
				   text="STOP",
				   command=self.stop_run, state=DISABLED)
		frame.stop_button.grid(column=2, row=2)

		frame.progress = ttk.Progressbar(frame, orient=HORIZONTAL, length=400, mode='determinate')
		frame.progress.grid(column=0, row=3, columnspan=3)

		frame.configure(background='#d8baa9')

	def add_images(self):
		
		new_files = filedialog.askopenfilenames(filetypes = (("tif files","*.tif"), ("all files","*.*")))
		new_files = list(set(new_files).difference(set(self.input_files)))

		self.input_files += new_files
		for filename in new_files: 
			self.file_display.file_box.insert(END, filename)
			self.update_log("Adding {}".format(filename))


	def add_directory(self):
		
		directory = filedialog.askdirectory()
		new_files = [directory + '/' + filename for filename in os.listdir(directory) if filename.endswith('.tif')]
		new_files = list(set(new_files).difference(set(self.input_files)))

		self.input_files += new_files

		for filename in new_files: 
			self.file_display.file_box.insert(END, filename)
			self.update_log("Adding {}".format(filename))


	def del_images(self):
		
		selected_files = [self.file_display.file_box.get(idx)\
							 for idx in self.file_display.file_box.curselection()]

		for filename in selected_files:
			index = self.input_files.index(filename)
			self.input_files.remove(filename)
			self.file_display.file_box.delete(index)
			self.update_log("Removing {}".format(filename))


	def create_image_display(self, notebook):

		#frame.grid(row=0, columnspan=3, sticky=(N,W,E,S))

		notebook.frame1 = ttk.Frame(notebook)
		notebook.add(notebook.frame1, text='Image')
		notebook.frame1.canvas = Canvas(notebook.frame1, width=650, height=550,
								scrollregion=(0,0,650,600))   # first page, which would get widgets gridded into it
		notebook.frame1.scrollbar = Scrollbar(notebook.frame1, orient=VERTICAL, 
											command=notebook.frame1.canvas.yview)
		notebook.frame1.scrollbar.pack(side=RIGHT,fill=Y)
		notebook.frame1.canvas['yscrollcommand'] = notebook.frame1.scrollbar.set
		notebook.frame1.canvas.pack(side = LEFT, fill = "both", expand = "yes")

		notebook.frame2 = ttk.Frame(notebook)
		notebook.add(notebook.frame2, text='Network')
		notebook.frame2.canvas = Canvas(notebook.frame2, width=650, height=550,
								scrollregion=(0,0,650,600))   # first page, which would get widgets gridded into it
		notebook.frame2.scrollbar = Scrollbar(notebook.frame2, orient=VERTICAL, 
											command=notebook.frame2.canvas.yview)
		notebook.frame2.scrollbar.pack(side=RIGHT,fill=Y)
		notebook.frame2.canvas['yscrollcommand'] = notebook.frame2.scrollbar.set
		notebook.frame2.canvas.pack(side = LEFT, fill = "both", expand = "yes")

		notebook.frame3 = ttk.Frame(notebook)
		notebook.add(notebook.frame3, text='Metrics')

		notebook.frame3.titles = ["Clustering", "Degree", "Linearity", "Coverage",
								"Fibre Waviness", "Network Waviness", "Solidity",
								"Pixel Anisotropy", "Region Anisotropy", "Image Anisotropy"]
		"""
		notebook.clustering = DoubleVar()
		notebook.degree = DoubleVar()
		notebook.linearity = DoubleVar()
		notebook.coverage = DoubleVar()
		notebook.fibre_waviness = DoubleVar()
		notebook.net_waviness = DoubleVar()
		notebook.solidity = DoubleVar()
		notebook.pix_anis = DoubleVar()
		notebook.region_anis = DoubleVar()
		notebook.img_anis = DoubleVar()

		notebook.frame3.cluster_title = Label(notebook.frame3, text="Clustering:")
		notebook.frame3.cluster = Label(notebook.frame3, textvariable=notebook.clustering)
		notebook.frame3.degree_title = Label(notebook.frame3, text="Degree:")
		notebook.frame3.degree = Label(notebook.frame3, textvariable=notebook.degree)
		notebook.frame3.linearity_title = Label(notebook.frame3, text="Linearity:")
		notebook.frame3.linearity = Label(notebook.frame3, textvariable=notebook.linearity)
		notebook.frame3.coverage_title = Label(notebook.frame3, text="Coverage:")
		notebook.frame3.coverage = Label(notebook.frame3, textvariable=notebook.coverage)
		notebook.frame3.f_wav_title = Label(notebook.frame3, text="Fibre Waviness:")
		notebook.frame3.f_wav = Label(notebook.frame3, textvariable=notebook.fibre_waviness)
		notebook.frame3.n_wav_title = Label(notebook.frame3, text="Network Waviness:")
		notebook.frame3.n_wav = Label(notebook.frame3, textvariable=notebook.net_waviness)
		notebook.frame3.solidity_title = Label(notebook.frame3, text="Solidity:")
		notebook.frame3.solidity = Label(notebook.frame3, textvariable=notebook.solidity)
		notebook.frame3.pix_anis_title = Label(notebook.frame3, text="Pixel Anisotropy:")
		notebook.frame3.pix_anis = Label(notebook.frame3, textvariable=notebook.pix_anis)
		notebook.frame3.region_anis_title = Label(notebook.frame3, text="Region Anisotropy:")
		notebook.frame3.region_anis = Label(notebook.frame3, textvariable=notebook.region_anis)
		notebook.frame3.img_anis_title = Label(notebook.frame3, text="Image Anisotropy:")
		notebook.frame3.img_anis = Label(notebook.frame3, textvariable=notebook.img_anis)

		notebook.frame3.cluster_title.grid(column=0, row=0)
		notebook.frame3.cluster.grid(column=1, row=0)
		notebook.frame3.degree_title.grid(column=0, row=1)
		notebook.frame3.degree.grid(column=1, row=1)
		notebook.frame3.linearity_title.grid(column=0, row=2)
		notebook.frame3.linearity.grid(column=1, row=2)
		notebook.frame3.coverage_title.grid(column=0, row=3)
		notebook.frame3.coverage.grid(column=1, row=3)
		notebook.frame3.f_wav_title.grid(column=0, row=4)
		notebook.frame3.f_wav.grid(column=1, row=4)
		notebook.frame3.n_wav_title.grid(column=0, row=5)
		notebook.frame3.n_wav.grid(column=1, row=5)
		notebook.frame3.solidity_title.grid(column=0, row=6)
		notebook.frame3.solidity.grid(column=1, row=6)
		notebook.frame3.pix_anis_title.grid(column=0, row=7)
		notebook.frame3.pix_anis.grid(column=1, row=7)
		notebook.frame3.region_anis_title.grid(column=0, row=8)
		notebook.frame3.region_anis.grid(column=1, row=8)
		notebook.frame3.img_anis_title.grid(column=0, row=9)
		notebook.frame3.img_anis.grid(column=1, row=9)
		"""

		notebook.frame3.fig, notebook.frame3.ax = plt.subplots(nrows=3, ncols=3, figsize=(3, 3), dpi=100)
		notebook.frame3.canvas = FigureCanvasTkAgg(notebook.frame3.fig, notebook.frame3)
		notebook.frame3.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

		notebook.frame3.toolbar = NavigationToolbar2Tk(notebook.frame3.canvas, notebook.frame3)
		notebook.frame3.toolbar.update()
		notebook.frame3.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

		notebook.frame4 = ttk.Frame(notebook)
		notebook.add(notebook.frame4, text='Log')
		notebook.frame4.text = Text(notebook.frame4, width=650, height=550)
		notebook.frame4.text.insert(END, self.Log)
		notebook.frame4.text.config(state=DISABLED)
		notebook.frame4.text.pack()


	def update_dashboard(self):

		for i, title in enumerate(self.image_display.frame3.titles[:9]):
			self.image_display.frame3.ax[i // 3][i % 3].clear()
			self.image_display.frame3.ax[i // 3][i % 3].set_title(title)
			self.image_display.frame3.ax[i // 3][i % 3].boxplot(self.database[title])
		self.image_display.frame3.canvas.draw()


	def update_log(self, text):

		self.image_display.frame4.text.config(state=NORMAL)
		self.Log += text + '\n'
		self.image_display.frame4.text.insert(END, text + '\n')
		self.image_display.frame4.text.config(state=DISABLED)


	def display_image(self, canvas, image):

		canvas.create_image(40, 20, image=image, anchor=NW)
		canvas.image = image
		canvas.pack(side = LEFT, fill = "both", expand = "yes")

		self.master.update_idletasks()


	def display_network(self, canvas, Aij):

		networks = []

		for i, component in enumerate(nx.connected_components(Aij)):
			subgraph = Aij.subgraph(component)
			if subgraph.number_of_nodes() > 3:
				networks.append(subgraph)

		for j, network in enumerate(networks):

			node_coord = np.stack((network.nodes[i]['xy'] for i in network.nodes()))

			mapping = dict(zip(network.nodes, np.arange(network.number_of_nodes())))
			network = nx.relabel_nodes(network, mapping)
			
			for n, node in enumerate(network.nodes):
				for m in list(network.adj[node]):
					canvas.create_line(node_coord[n][1]+40, node_coord[n][0]+20,
								       node_coord[m][1]+40, node_coord[m][0]+20,
										fill="red", width=3)

	def display_notebook(self):

		selected_file = [self.file_display.file_box.get(idx)\
							 for idx in self.file_display.file_box.curselection()][0]

		image_name = selected_file.split('/')[-1]
		image_path = '/'.join(selected_file.split('/')[:-1])
		fig_name = ut.check_file_name(image_name, extension='tif')
		data_dir = image_path + '/data/'

		image_tk = ImageTk.PhotoImage(Image.open(selected_file))
		self.display_image(self.image_display.frame1.canvas, image_tk)
		self.update_log("Displaying image {}".format(fig_name))

		try:
			Aij = nx.read_gpickle(data_dir + fig_name + ".pkl")
			self.display_image(self.image_display.frame2.canvas, image_tk)
			self.display_network(self.image_display.frame2.canvas, Aij)
			self.update_log("Displaying network for {}".format(fig_name))
		except IOError:
			self.update_log("Unable to display network for {}".format(fig_name))

		"""
		try:
			self.image_display.metrics = ut.load_npy(data_dir + fig_name)
			self.image_display.clustering.set(self.image_display.metrics[0])
			self.image_display.degree.set(self.image_display.metrics[1])
			self.image_display.linearity.set(self.image_display.metrics[2])
			self.image_display.coverage.set(self.image_display.metrics[3])
			self.image_display.fibre_waviness.set(self.image_display.metrics[4])
			self.image_display.net_waviness.set(self.image_display.metrics[5])
			self.image_display.solidity.set(self.image_display.metrics[6])
			self.image_display.pix_anis.set(self.image_display.metrics[7])
			self.image_display.region_anis.set(self.image_display.metrics[8])
			self.image_display.img_anis.set(self.image_display.metrics[9])

		except IOError: pass
		"""
		self.master.update_idletasks()


	def generate_db(self):

		database_array = np.empty((0, 10), dtype=float)
		database_index = []

		for i, input_file_name in enumerate(self.input_files):

			image_name = input_file_name.split('/')[-1]
			image_path = '/'.join(input_file_name.split('/')[:-1])
			data_dir = image_path + '/data/'
			metric_name = data_dir + ut.check_file_name(image_name, extension='tif')
			
			self.update_log("Loading metrics for {}".format(input_file_name))

			try: 
				database_array = np.concatenate((database_array, 
											np.expand_dims(ut.load_npy(metric_name), axis=0)))
				database_index.append(input_file_name)
			except IOError:
				self.update_log("{} database not generated - check log")
				return

		print(database_array.shape)

		self.database = pd.DataFrame(data=database_array, columns=self.image_display.frame3.titles,
					 index = self.input_files)

		self.update_dashboard()


	def save_db(self):

		db_filename = filedialog.asksaveasfilename(defaultextension=".pkl")
		self.database.to_pickle(db_filename)
		self.update_log("Saving Database file {}".format(db_filename))


	def write_run(self):

		self.file_display.run_button.config(state=DISABLED)	
		self.file_display.stop_button.config(state=NORMAL)
		self.file_display.progress['maximum'] = len(self.input_files)

		"""
		ow_metric = [self.ow_metric.get()] * len(self.input_files)
		ow_network = [self.ow_network.get()] * len(self.input_files)
		queue = [self.queue] * len(self.input_files)
		args = zip(ow_metric, ow_network, queue)
		for arg in args: print(arg)

		self.process = Pool(self.n_proc)
		self.process.daemon = True
		self.process.map(image_analysis, args)
		"""

		self.process = Process(target=image_analysis, args=(self.input_files, self.ow_metric.get(),
														self.ow_network.get(), self.queue))
		self.process.daemon = True
		self.process.start()
		self.process_check()


	def process_check(self):
		"""
		Check if there is something in the queue
		"""
		self.queue_check()

		if self.process.exitcode is None:
			self.master.after(500, self.process_check)
		else: 
			self.file_display.progress['value'] = 0
			self.file_display.run_button.config(state=NORMAL)
			self.file_display.stop_button.config(state=DISABLED)
			self.generate_db()
			if self.save_db.get(): self.save_db()


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
		self.process.terminate()


def image_analysis(input_files, ow_metric, ow_network, queue):

	for input_file_name in input_files:

		image_name = input_file_name.split('/')[-1]
		image_path = '/'.join(input_file_name.split('/')[:-1])
		fig_name = ut.check_file_name(image_name, extension='tif')

		try:
			analyse_image(image_path, input_file_name, ow_metric=ow_metric, 
					ow_network=ow_network, sigma=0.5, snr_thresh=0.18)
			queue.put("Analysis of {} complete".format(input_file_name))

		except NoiseError as err: queue.put("{} {}".format(err.message, input_file_name))


N_PROC = os.cpu_count() - 1

root = Tk()
GUI = imagecol_gui(root, N_PROC)

root.mainloop()
