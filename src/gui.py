import matplotlib
matplotlib.use("Agg")

import os, sys, time
from tkinter import *
from tkinter import ttk, filedialog
import queue, threading
from multiprocessing import Pool, Process, JoinableQueue, Queue, current_process

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from PIL import ImageTk, Image
import networkx as nx
import numpy as np
import pandas as pd

from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter

from skimage import img_as_float, measure
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.filters import threshold_otsu
from skimage.color import gray2rgb, label2rgb
from skimage.restoration import (estimate_sigma, denoise_tv_chambolle, denoise_bilateral)

from main import analyse_image
import utilities as ut
from utilities import NoiseError
from image_tools import load_image, draw_network, network_area

class imagecol_gui:

	def __init__(self, master, n_proc, n_thread):

		"Set file locations"
		self.source_dir = os.path.dirname(os.path.realpath(__file__))
		self.pyfibre_dir = self.source_dir[:self.source_dir.rfind(os.path.sep)]
		self.current_dir = os.getcwd()

		"Initiatise program log, queue and input file list"
		self.Log = "Initiating PyFibre GUI\n"
		self.queue = Queue()
		self.input_files = []
		self.n_proc = n_proc
		self.n_thread = n_thread

		"Initialise option variables"
		self.ow_metric = IntVar()
		self.ow_network = IntVar()
		self.save_db = IntVar()
		self.sigma = DoubleVar()
		self.sigma.set(0.5)
		self.p0 = IntVar()
		self.p0.set(1)
		self.p1 = IntVar()
		self.p1.set(98)
		self.n = IntVar()
		self.n.set(5)
		self.m = IntVar()
		self.m.set(35)

		"Define GUI objects"
		self.master = master
		self.master.geometry("1230x620")
		self.master.configure(background='#d8baa9')
		self.master.protocol("WM_DELETE_WINDOW", lambda: quit())

		self.title = Frame(self.master)
		self.create_title(self.title)
		self.title.place(bordermode=OUTSIDE, height=200, width=300)

		self.options = Frame(self.master)
		self.options.configure(background='#d8baa9')
		self.options.options_button = Button(self.options, width=15,
				   text="Options",
				   command=self.create_options)
		self.options.options_button.pack()
		self.options.place(x=300, y=1, height=200, width=250)

		self.file_display = Frame(self.master)
		self.create_file_display(self.file_display)
		self.file_display.place(x=5, y=220, height=600, width=545)

		self.image_display = ttk.Notebook(self.master)
		self.create_image_display(self.image_display)
		self.master.bind('<Double-1>', lambda e: self.display_notebook())
		self.image_display.place(x=550, y=10, width=675, height=600)


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

		"Initialise option parameters"
		frame = Toplevel(self.master)
		frame.title('PyFibre - Options')
		frame.geometry('310x500-100+40')

		frame.title_sigma = Label(frame, text="Gaussian Std Dev (pix)")
		frame.title_sigma.configure(background='#d8baa9')
		frame.sigma = Scale(frame, from_=0, to=10, tickinterval=1, resolution=0.1, 
						length=300, orient=HORIZONTAL, variable=self.sigma)#,text='Low Clip Intensity (%)')

		frame.title_sigma.grid(column=0, row=2, rowspan=1)
		frame.sigma.grid(column=0, row=3, sticky=(N,W,E,S))

		frame.title_p0 = Label(frame, text="Low Clip Intensity (%)")
		frame.title_p0.configure(background='#d8baa9')
		frame.p0 = Scale(frame, from_=0, to=100, tickinterval=10, 
						length=300, orient=HORIZONTAL, variable=self.p0)#,text='Low Clip Intensity (%)')
		frame.title_p1 = Label(frame, text="High Clip Intensity (%)")
		frame.title_p1.configure(background='#d8baa9')
		frame.p1 = Scale(frame, from_=0, to=100,tickinterval=10, 
							length=300, orient=HORIZONTAL, variable=self.p1)#, text='High Clip Intensity (%)')

		frame.title_p0.grid(column=0, row=4, rowspan=1)
		frame.p0.grid(column=0, row=5)
		frame.title_p1.grid(column=0, row=6, rowspan=1)
		frame.p1.grid(column=0, row=7)

		frame.title_n = Label(frame, text="NL-Mean Neighbourhood 1 (pix)")
		frame.title_n.configure(background='#d8baa9')
		frame.n = Scale(frame, from_=0, to=100, tickinterval=10, 
						length=300, orient=HORIZONTAL, variable=self.n)
		frame.title_m = Label(frame, text="NL-Mean Neighbourhood 2 (pix)")
		frame.title_m.configure(background='#d8baa9')
		frame.m = Scale(frame, from_=0, to=100,tickinterval=10,
						length=300, orient=HORIZONTAL, variable=self.m)

		frame.title_n.grid(column=0, row=8, rowspan=1)
		frame.n.grid(column=0, row=9)
		frame.title_m.grid(column=0, row=10, rowspan=1)
		frame.m.grid(column=0, row=11)

		frame.chk_anis = Checkbutton(frame, text="o/w metrics", variable=self.ow_metric)
		frame.chk_anis.configure(background='#d8baa9')
		frame.chk_anis.grid(column=0, row=12, sticky=(N,W,E,S))
		#frame.chk_anis.pack(side=LEFT)

		frame.chk_graph = Checkbutton(frame, text="o/w graph", variable=self.ow_network)
		frame.chk_graph.configure(background='#d8baa9')
		frame.chk_graph.grid(column=0, row=13, sticky=(N,W,E,S))
		#frame.chk_graph.pack(side=LEFT)

		frame.chk_db = Checkbutton(frame, text="Save Database", variable=self.save_db)
		frame.chk_db.configure(background='#d8baa9')
		frame.chk_db.grid(column=0, row=14, sticky=(N,W,E,S))

		frame.configure(background='#d8baa9')


	def create_file_display(self, frame):

		frame.select_im_button = Button(frame, width=12,
				   text="Load Files",
				   command=self.add_images)
		frame.select_im_button.grid(column=0, row=0)

		frame.select_dir_button = Button(frame, width=12,
				   text="Load Folder",
				   command=self.add_directory)
		frame.select_dir_button.grid(column=1, row=0)

		frame.key = Entry(frame, width=10)
		frame.key.configure(background='#d8baa9')
		frame.key.grid(column=3, row=0, sticky=(N,W,E,S))

		frame.select_dir_button = Button(frame, width=12,
				   text="Filter",
				   command=lambda : self.del_images([filename for filename in self.input_files \
							if (filename.find(frame.key.get()) == -1)]))
		frame.select_dir_button.grid(column=2, row=0)


		frame.file_box = Listbox(frame, height=20, width=50, selectmode="extended")
		frame.file_box.grid(column=0, row=1, columnspan=5, sticky=(N,W,E,S))

		frame.delete_im_button = Button(frame, width=12,
				   text="Delete",
				   command=lambda : self.del_images([self.file_display.file_box.get(idx)\
							 for idx in self.file_display.file_box.curselection()]))
		frame.delete_im_button.grid(column=4, row=0)

		frame.scrollbar = ttk.Scrollbar(frame, orient=VERTICAL, command=frame.file_box.yview)
		frame.scrollbar.grid(column=5, row=1, sticky=(N,S))
		frame.file_box['yscrollcommand'] = frame.scrollbar.set

		#frame.grid_columnconfigure(1, weight=1)
		#frame.grid_rowconfigure(1, weight=1)

		frame.run_button = Button(frame, width=40,
				   text="GO",
				   command=self.write_run)
		frame.run_button.grid(column=0, row=2, columnspan=3)

		frame.stop_button = Button(frame, width=20,
				   text="STOP",
				   command=self.stop_run, state=DISABLED)
		frame.stop_button.grid(column=2, row=2, columnspan=3)

		frame.progress = ttk.Progressbar(frame, orient=HORIZONTAL, length=400, mode='determinate')
		frame.progress.grid(column=0, row=3, columnspan=5)

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
		new_files = [directory + '/' + filename for filename in os.listdir(directory) \
				 if filename.endswith('.tif')]
		new_files = list(set(new_files).difference(set(self.input_files)))

		self.input_files += new_files

		for filename in new_files: 
			self.file_display.file_box.insert(END, filename)
			self.update_log("Adding {}".format(filename))


	def del_images(self, file_list):

		for filename in file_list:
			index = self.input_files.index(filename)
			self.input_files.remove(filename)
			self.file_display.file_box.delete(index)
			self.update_log("Removing {}".format(filename))


	def create_image_display(self, notebook):

		#frame.grid(row=0, columnspan=3, sticky=(N,W,E,S))

		notebook.image_tab = ttk.Frame(notebook)
		notebook.add(notebook.image_tab, text='Image')
		notebook.image_tab.canvas = Canvas(notebook.image_tab, width=675, height=550,
								scrollregion=(0,0,675,600))  
		notebook.image_tab.scrollbar = Scrollbar(notebook.image_tab, orient=VERTICAL, 
							command=notebook.image_tab.canvas.yview)
		notebook.image_tab.scrollbar.pack(side=RIGHT,fill=Y)
		notebook.image_tab.canvas['yscrollcommand'] = notebook.image_tab.scrollbar.set
		notebook.image_tab.canvas.pack(side = LEFT, fill = "both", expand = "yes")

		notebook.network_tab = ttk.Frame(notebook)
		notebook.add(notebook.network_tab, text='Network')
		notebook.network_tab.canvas = Canvas(notebook.network_tab, width=675, height=550,
								scrollregion=(0,0,675,600))  
		notebook.network_tab.scrollbar = Scrollbar(notebook.network_tab, orient=VERTICAL, 
							command=notebook.network_tab.canvas.yview)
		notebook.network_tab.scrollbar.pack(side=RIGHT,fill=Y)
		notebook.network_tab.canvas['yscrollcommand'] = notebook.network_tab.scrollbar.set
		notebook.network_tab.canvas.pack(side = LEFT, fill = "both", expand = "yes")

		notebook.segment_tab = ttk.Frame(notebook)
		notebook.add(notebook.segment_tab, text='Segment')
		notebook.segment_tab.canvas = Canvas(notebook.segment_tab, width=675, height=550,
								scrollregion=(0,0,675,600))  
		notebook.segment_tab.scrollbar = Scrollbar(notebook.segment_tab, orient=VERTICAL, 
							command=notebook.segment_tab.canvas.yview)
		notebook.segment_tab.scrollbar.pack(side=RIGHT,fill=Y)
		notebook.segment_tab.canvas['yscrollcommand'] = notebook.segment_tab.scrollbar.set
		notebook.segment_tab.canvas.pack(side = LEFT, fill = "both", expand = "yes")

		notebook.metric_tab = ttk.Frame(notebook)
		notebook.add(notebook.metric_tab, text='Metrics')

		notebook.metric_tab.metric_dict = {'SDI' : {"info" : "Fourier spectrum SDI of total image", "metric" : DoubleVar()},
										'Entropy' : {"info" : "Average Shannon entropy of segmented image", "metric" : DoubleVar()},
										'Pixel Anisotropy' : {"info" : "Average anisotropy of all pixels in total image", "metric" : DoubleVar()},
										'Anisotropy' : {"info" : "Anisotropy of total image", "metric" : DoubleVar()}, 
										'Coverage' : {"info" : "Ratio of total image covered by collagen fibres", "metric" : DoubleVar()}, 
										#'Local Pixel Anisotropy' : {"info" : "Average anisotropy of all pixels in segmented image", "metric" : DoubleVar()},
										#'Local Anisotropy' : {"info" : "Average Anisotropy of segented image", "metric" : DoubleVar()}, 
										'Linearity' : {"info" : "Average segment shape linearity", "metric" : DoubleVar()}, 
										'Eccentricity' : {"info" : "Average segment shape eccentricity", "metric" : DoubleVar()},
										'Density' : {"info" : "Average segment density", "metric" : DoubleVar()},
										'Network Waviness' : {"info" : "Average fibre network fibre waviness", "metric" : DoubleVar()},
										'Network Degree' : {"info" : "Average fibre network number of edges per node", "metric" : DoubleVar()},
										'Network Centrality' : {"info" : "Average fibre network centrality", "metric" : DoubleVar()},
										'Network Connectivity' : {"info" : "Average fibre network connectivity", "metric" : DoubleVar()},
										'Network Local Efficiency' : {"info" : "Average fibre network local efficiency", "metric" : DoubleVar()}
										}

		notebook.metric_tab.titles = list(notebook.metric_tab.metric_dict.keys())

		notebook.metrics = [DoubleVar() for i in range(len(notebook.metric_tab.titles))]
		notebook.metric_tab.headings = []
		notebook.metric_tab.info = []
		notebook.metric_tab.metrics = []

		for i, metric in enumerate(notebook.metric_tab.titles):
			notebook.metric_tab.headings += [Label(notebook.metric_tab, text="{}:".format(metric))]
			notebook.metric_tab.info += [Label(notebook.metric_tab, text=notebook.metric_tab.metric_dict[metric]["info"])]
			notebook.metric_tab.metrics += [Label(notebook.metric_tab, textvariable=notebook.metric_tab.metric_dict[metric]["metric"])]
			notebook.metric_tab.headings[i].grid(column=0, row=i)
			notebook.metric_tab.info[i].grid(column=1, row=i)
			notebook.metric_tab.metrics[i].grid(column=2, row=i)

		notebook.log_tab = ttk.Frame(notebook)
		notebook.add(notebook.log_tab, text='Log')
		notebook.log_tab.text = Text(notebook.log_tab, width=675, height=550)
		notebook.log_tab.text.insert(END, self.Log)
		notebook.log_tab.text.config(state=DISABLED)

		notebook.log_tab.scrollbar = Scrollbar(notebook.log_tab, orient=VERTICAL, 
							command=notebook.log_tab.text.yview)
		notebook.log_tab.scrollbar.pack(side=RIGHT,fill=Y)
		notebook.log_tab.text['yscrollcommand'] = notebook.log_tab.scrollbar.set

		notebook.log_tab.text.pack()
		

		#notebook.BFrame.configure(background='#d8baa9')


	def update_log(self, text):

		self.image_display.log_tab.text.config(state=NORMAL)
		self.Log += text + '\n'
		self.image_display.log_tab.text.insert(END, text + '\n')
		self.image_display.log_tab.text.config(state=DISABLED)


	def display_image(self, canvas, image):

		canvas.create_image(40, 20, image=image, anchor=NW)
		canvas.image = image
		canvas.pack(side = LEFT, fill = "both", expand = "yes")

		self.master.update_idletasks()


	def display_network(self, canvas, image, Aij):

		self.display_image(canvas, image)

		networks = []

		for i, component in enumerate(nx.connected_components(Aij)):
			subgraph = Aij.subgraph(component)
			if subgraph.number_of_nodes() > 3:
				networks.append(subgraph)
				

		for j, network in enumerate(networks):

			node_coord = [network.nodes[i]['xy'] for i in network.nodes()]
			node_coord = np.stack(node_coord)

			mapping = dict(zip(network.nodes, np.arange(network.number_of_nodes())))
			network = nx.relabel_nodes(network, mapping)
			
			for n, node in enumerate(network.nodes):
				for m in list(network.adj[node]):
					canvas.create_line(node_coord[n][1] + 40, node_coord[n][0] + 20,
								       node_coord[m][1] + 40, node_coord[m][0] + 20,
										fill="red", width=2)


	def display_segments(self, canvas, image, Aij):

		networks = []
		image *= 100 / image.max()
		label_image = np.zeros(image.shape, dtype=int)

		for i, component in enumerate(nx.connected_components(Aij)):
			subgraph = Aij.subgraph(component)
			if subgraph.number_of_nodes() > 3:
				networks.append(subgraph)
				label_image = draw_network(subgraph, label_image, i + 1)

		segmented_image = np.zeros(image.shape, dtype=int)
		rect = []

		"Measure pixel areas of each segment"
		for i, region in enumerate(measure.regionprops(label_image)):
			minr, minc, maxr, maxc = region.bbox
			indices = np.mgrid[minr:maxr, minc:maxc]
			area, segment = network_area(region.image)

			if region.area >= 4E-4 * image.size:
				segmented_image[(indices[0], indices[1])] += segment * (i + 1)

		image_label_overlay = label2rgb(segmented_image, image=image, bg_label=0,
										image_alpha=0.9, alpha=0.9, bg_color=(0, 0, 0))
		image_label_overlay *= 4 * 255.9999 / image_label_overlay.max()
		image_pil = Image.fromarray(image_label_overlay.astype('uint8'))
		image_tk = ImageTk.PhotoImage(image_pil)

		self.display_image(canvas, image_tk)


	def display_notebook(self):

		selected_file = [self.file_display.file_box.get(idx)\
							 for idx in self.file_display.file_box.curselection()][0]

		image_name = selected_file.split('/')[-1]
		image_path = '/'.join(selected_file.split('/')[:-1])
		fig_name = ut.check_file_name(image_name, extension='tif')
		data_dir = image_path + '/data/'

		image_numpy = 4 * 255.999 * load_image(selected_file)
		image_tk = ImageTk.PhotoImage(Image.fromarray(image_numpy.astype('uint8')))
		self.display_image(self.image_display.image_tab.canvas, image_tk)
		self.update_log("Displaying image {}".format(fig_name))

		try:
			Aij = nx.read_gpickle(data_dir + fig_name + "_network.pkl")
			self.display_network(self.image_display.network_tab.canvas, image_tk, Aij)
			self.display_segments(self.image_display.segment_tab.canvas, image_numpy, Aij)
			self.update_log("Displaying network and segments for {}".format(fig_name))
		except IOError:
			self.update_log("Unable to display network and segments for {}".format(fig_name))

		#"""
		try:
			#loaded_metrics = ut.load_npy(data_dir + fig_name)
			loaded_metrics = pd.read_pickle('{}_metric.pkl'.format(data_dir + fig_name)).iloc[0]
			for i, metric in enumerate(self.image_display.metric_tab.titles):
				self.image_display.metric_tab.metric_dict[metric]["metric"].set(loaded_metrics[metric])
			self.update_log("Displaying metrics for {}".format(fig_name))

		except IOError:
			self.update_log("Unable to display metrics for {}".format(fig_name))
			for i, metric in enumerate(self.image_display.metric_tab.titles):
				self.image_display.metric_tab.metric_dict[metric]["metric"].set(0)
		#"""
		self.master.update_idletasks()


	def generate_db(self):

		global_database = pd.DataFrame()
		segment_database = pd.DataFrame()

		for i, input_file_name in enumerate(self.input_files):

			image_name = input_file_name.split('/')[-1]
			image_path = '/'.join(input_file_name.split('/')[:-1])
			data_dir = image_path + '/data/'
			metric_name = data_dir + ut.check_file_name(image_name, extension='tif')
			
			self.update_log("Loading metrics for {}".format(metric_name))

			try: 
				data = pd.read_pickle('{}_metric.pkl'.format(metric_name))
				global_database = pd.concat([global_database, data.iloc[:1]], sort=True)
				segment_database = pd.concat([segment_database, data.iloc[1:]], sort=True)

			except (ValueError, IOError):
				self.update_log(f"{input_file_name} database not imported - skipping")

		self.global_database = global_database
		self.segment_database = segment_database

		#self.update_dashboard()


	def save_database(self):

		db_filename = filedialog.asksaveasfilename()
		db_filename = ut.check_file_name(db_filename, extension='pkl')
		db_filename = ut.check_file_name(db_filename, extension='xls')

		self.global_database.to_pickle(db_filename + '_global.pkl')
		self.global_database.to_excel(db_filename + '_global.xls')
		self.segment_database.to_pickle(db_filename + '_segment.pkl')
		self.segment_database.to_excel(db_filename + '_segment.xls')

		self.update_log("Saving Database files {}".format(db_filename))


	def write_run(self):

		self.file_display.run_button.config(state=DISABLED)	
		self.file_display.stop_button.config(state=NORMAL)
		self.file_display.progress['maximum'] = len(self.input_files)

		#"""Multi Processor version
		proc_count = np.min((self.n_proc, len(self.input_files)))
		batch_files = np.array_split(self.input_files, proc_count)
		self.processes = []
		for batch in batch_files:
			process = Process(target=image_analysis, 
					args=(batch, 
					(self.p0.get(), self.p1.get()),
					(self.n.get(), self.m.get()),
					self.sigma.get(),
					self.ow_metric.get(), self.ow_network.get(), 
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


def image_analysis(input_files, p_intensity, p_denoise, sigma, ow_metric, ow_network, queue, threads):

	for input_file_name in input_files:

		image_path = '/'.join(input_file_name.split('/')[:-1])

		try:
			analyse_image(input_file_name, image_path,
					scale=1, p_intensity=p_intensity,
					p_denoise=p_denoise, sigma=sigma, 
					ow_metric=ow_metric, ow_network=ow_network,
					threads=threads)
			queue.put("Analysis of {} complete".format(input_file_name))

		except NoiseError as err: queue.put("{} {}".format(err.message, input_file_name))


N_PROC = 1#os.cpu_count() - 1
N_THREAD = 8

root = Tk()
GUI = imagecol_gui(root, N_PROC, N_THREAD)

root.mainloop()
