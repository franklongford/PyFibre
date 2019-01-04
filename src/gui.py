import os, sys, time
from tkinter import *
from tkinter import ttk, filedialog
import queue, threading
from multiprocessing import Process, JoinableQueue, Queue

from PIL import ImageTk, Image
import networkx as nx
import numpy as np

from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter

from skimage import img_as_float
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from skimage.color import gray2rgb
from skimage.restoration import (estimate_sigma, denoise_tv_chambolle, denoise_bilateral)

from main import *
import utilities as ut
from anisotropy import analyse_image

class imagecol_gui:

	def __init__(self, master):

		self.input_files = []
		self.dir_path = os.path.dirname(os.path.realpath(__file__))
		self.current_dir = os.getcwd()

		self.master = master
		self.master.title("PyFibre - Python Fibrous Image Toolkit")
		img = ImageTk.PhotoImage(file=self.dir_path + '/icon.ico')
		self.master.tk.call('wm', 'iconphoto', self.master._w, img)
		self.master.geometry("1100x620")
		#self.master.protocol('WM_DELETE_WINDOW', self.exit_app)

		self.title = Frame(self.master)
		self.create_title(self.title)
		#self.title.pack()
		self.title.place(bordermode=OUTSIDE, height=200, width=300)

		self.frame_options = Frame(self.master)
		self.create_options(self.frame_options)
		#self.frame_options.pack()
		self.frame_options.place(x=300, y=1, height=200, width=200)

		self.frame_display = Frame(self.master)
		self.display_image_files(self.frame_display)
		#self.frame_display.pack()
		self.frame_display.place(x=5, y=220, height=600, width=500)

		self.image_display = ttk.Notebook(self.master)
		self.create_notebook(self.image_display)
		#self.image_display.pack()
		self.image_display.place(x=450, y=10, width=625, height=600)

		self.master.bind('<Double-1>', lambda e: self.display_notebook())

	def exit_app(self): 

		self.process.join()
		sys.exit()


	def create_title(self, frame):

		image = Image.open(self.dir_path + '/icon.ico')
		image = image.resize((300,200))
		image_tk = ImageTk.PhotoImage(image)
		frame.text_title = Label(frame, image=image_tk)
		frame.image = image_tk
		frame.text_title.pack(side = TOP, fill = "both", expand = "yes")

	def create_options(self, frame):

		self.ow_anis = IntVar()
		frame.chk_anis = Checkbutton(frame, text="o/w anisotropy", variable=self.ow_anis)
		frame.chk_anis.grid(column=0, row=0, sticky=(N,W,E,S))
		#frame.chk_anis.pack(side=LEFT)

		self.ow_graph = IntVar()
		frame.chk_graph = Checkbutton(frame, text="o/w graph", variable=self.ow_graph)
		frame.chk_graph.grid(column=0, row=1, sticky=(N,W,E,S))
		#frame.chk_graph.pack(side=LEFT)


	def display_image_files(self, frame):

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

		frame.image_box = Listbox(frame, height=20, width=40, selectmode="extended")
		frame.image_box.grid(column=0, row=1, columnspan=3, sticky=(N,W,E,S))

		frame.scrollbar = ttk.Scrollbar(frame, orient=VERTICAL, command=frame.image_box.yview)
		frame.scrollbar.grid(column=3, row=1, sticky=(N,S))
		frame.image_box['yscrollcommand'] = frame.scrollbar.set

		#frame.grid_columnconfigure(1, weight=1)
		#frame.grid_rowconfigure(1, weight=1)

		frame.run_button = Button(frame, width=30,
				   text="GO",
				   command=self.write_run)
		frame.run_button.grid(column=0, row=2, columnspan=2)

		frame.quit_button = Button(frame, width=15,
				   text="QUIT",
				   command=self.master.quit)
		frame.quit_button.grid(column=2, row=2)

		frame.progress = ttk.Progressbar(frame, orient=HORIZONTAL, length=300, mode='determinate')
		frame.progress.grid(column=0, row=3, columnspan=3)

		frame.pack()


	def add_images(self):
		
		new_files = filedialog.askopenfilenames(filetypes = (("tif files","*.tif"), ("all files","*.*")))
		new_files = list(set(new_files).difference(set(self.input_files)))

		self.input_files += new_files
		for filename in new_files: self.frame_display.image_box.insert(END, filename)


	def add_directory(self):
		
		directory = filedialog.askdirectory()
		new_files = [directory + '/' + filename for filename in os.listdir(directory) if filename.endswith('.tif')]
		new_files = list(set(new_files).difference(set(self.input_files)))

		self.input_files += new_files

		for filename in new_files: self.frame_display.image_box.insert(END, filename)


	def del_images(self):
		
		selected_files = [self.frame_display.image_box.get(idx)\
							 for idx in self.frame_display.image_box.curselection()]

		for filename in selected_files:
			index = self.input_files.index(filename)
			self.input_files.remove(filename)
			self.frame_display.image_box.delete(index)


	def create_notebook(self, notebook):

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

		selected_file = [self.frame_display.image_box.get(idx)\
							 for idx in self.frame_display.image_box.curselection()][0]

		image_name = selected_file.split('/')[-1]
		image_path = '/'.join(selected_file.split('/')[:-1])
		fig_name = ut.check_file_name(image_name, extension='tif')
		data_dir = image_path + '/data/'

		image_tk = ImageTk.PhotoImage(Image.open(selected_file))
		self.display_image(self.image_display.frame1.canvas, image_tk)
		try:
			Aij = nx.read_gpickle(data_dir + fig_name + ".pkl")
			self.display_image(self.image_display.frame2.canvas, image_tk)
			self.display_network(self.image_display.frame2.canvas, Aij)
		except IOError: pass

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
			print("Updated Metrics")

		except IOError: pass

		self.master.update_idletasks()

	def write_run(self):

		self.frame_display.run_button.config(state=DISABLED)
		self.task_queue = queue.Queue()
		self.result_queue = queue.Queue()

		self.process = Consumer(self.task_queue, self.result_queue)
		self.process.daemon = True
		self.process.start()

		self.frame_display.progress['value'] = 0
		snr_thresh = 2.0
		values = 100 // len(self.input_files)

		for i, input_file_name in enumerate(self.input_files):
			self.frame_display.progress['value'] = (i + 1) * values
			self.frame_display.progress.update()
			
			#self.process = Process(target=self.image_analysis, args=(input_file_name))
			self.task_queue.put(self.image_analysis(input_file_name))
	
		print(self.task_queue.qsize())
		self.task_queue.put(None)

		while not self.result_queue.empty():
			print(self.result_queue.get())

		self.frame_display.run_button.config(state=NORMAL)
		print("Run Complete")


	def image_analysis(self, input_file_name):

		image_name = input_file_name.split('/')[-1]
		image_path = '/'.join(input_file_name.split('/')[:-1])
		fig_name = ut.check_file_name(image_name, extension='tif')

		analyse_image(image_path, input_file_name, ow_anis=self.ow_anis.get(), 
				ow_graph=self.ow_graph.get(), sigma=0.5, mode='test', snr_thresh=0.18)


	def process_queue(self):

		if self.process.is_alive():
			print("Still running") 
			self.master.after(100, self.process_queue)
		else:
			print("Process {} complete".format(self.process))
			try:
				msg = self.queue.get(0)
				print(msg)
			except queue.Empty: print("queue is empty")
		


class Consumer(threading.Thread):
    
	def __init__(self, task_queue, result_queue):
		threading.Thread.__init__(self)
		self.task_queue = task_queue
		self.result_queue = result_queue

	def run(self):
		proc_name = self.name
		while True:
			next_task = self.task_queue.get()
			if next_task is None:
				# Poison pill means shutdown
				print('%s: Exiting' % proc_name)
				self.task_queue.task_done()
				break
			print('%s: %s' % (proc_name, next_task))
			answer = next_task()
			self.task_queue.task_done()
			self.result_queue.put(answer)
		return


side = LEFT
anchor = W

root = Tk()
GUI = imagecol_gui(root)

root.mainloop()
