import os, sys
from tkinter import *
from tkinter import ttk, filedialog

from PIL import ImageTk, Image
import networkx as nx
import numpy as np

from main import *
import utilities as ut
from anisotropy import analyse_image

class imagecol_gui:

	def __init__(self, master):

		self.input_files = []

		self.master = master
		self.master.title("FITI - Fibrous Tissue Image Toolkit")

		self.text_title = Text(self.master, height=18, width=100)
		self.text_title.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
		self.text_title.tag_configure('big', font=('Verdana', 10, 'bold'))
		self.text_title.tag_configure('color', foreground='#476042', 
						font=('Tempus Sans ITC', 12, 'bold'))
		self.text_title.insert(END, ut.logo(), 'big')
		self.text_title.pack()

		self.frame_options = Frame(master)

		self.ow_anis = IntVar()
		self.chk_anis = Checkbutton(self.frame_options, text="overwrite anisotropy", variable=self.ow_anis)
		#self.chk_anis.grid(column=0, row=0, sticky=(N,W,E,S))
		self.chk_anis.pack(side=LEFT)

		self.ow_graph = IntVar()
		self.chk_graph = Checkbutton(self.frame_options, text="overwrite graph", variable=self.ow_graph)
		#self.chk_graph.grid(column=0, row=1, sticky=(N,W,E,S))
		self.chk_graph.pack(side=RIGHT)

		self.frame_options.pack()

		self.frame_display = Frame(self.master)
		self.display_image_files(self.frame_display)
		self.frame_display.pack()

		self.image_display = Frame(self.master)
		self.image_display.canvas = Canvas(self.image_display, width=600, height=550,
							scrollregion=(0,0,650,600))
		self.image_display.scrollbar = Scrollbar(self.image_display, orient=VERTICAL, command=self.image_display.canvas.yview)
		self.image_display.scrollbar.pack(side=RIGHT,fill=Y)
		self.image_display.canvas['yscrollcommand'] = self.image_display.scrollbar.set
		self.image_display.pack()


	def display_image_files(self, frame):

		frame.select_im_button = Button(frame, width=40,
				   text="Select files",
				   command=self.add_images)
		frame.select_im_button.grid(column=0, row=0)

		frame.select_dir_button = Button(frame, width=40,
				   text="Select directory",
				   command=self.add_directory)
		frame.select_dir_button.grid(column=1, row=0)

		frame.delete_im_button = Button(frame, width=40,
				   text="Delete",
				   command=self.del_images)
		frame.delete_im_button.grid(column=2, row=0)

		frame.image_box = Listbox(frame, width=80, selectmode="extended")
		frame.image_box.grid(column=0, row=1, columnspan=3, sticky=(N,W,E,S))

		frame.scrollbar = ttk.Scrollbar(frame, orient=VERTICAL, command=frame.image_box.yview)
		frame.scrollbar.grid(column=3, row=1, sticky=(N,S))
		frame.image_box['yscrollcommand'] = frame.scrollbar.set

		#frame.grid_columnconfigure(1, weight=1)
		#frame.grid_rowconfigure(1, weight=1)

		frame.run_button = Button(frame, width=80,
				   text="GO",
				   command=self.write_run)
		frame.run_button.grid(column=0, row=2, columnspan=2)

		frame.quit_button = Button(frame, width=40,
				   text="QUIT",
				   command=self.master.quit)
		frame.quit_button.grid(column=2, row=2)

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
		
		removed_files = [self.frame_display.image_box.get(idx) for idx in self.frame_display.image_box.curselection()]

		for filename in removed_files:
			index = self.input_files.index(filename)
			self.input_files.remove(filename)
			self.frame_display.image_box.delete(index)


	def display_image_box(self, frame, image):

		#frame.grid(row=0, columnspan=3, sticky=(N,W,E,S))
		frame.canvas.create_image(40, 20, image=image, anchor=NW)
		frame.canvas.image = image
		frame.canvas.pack(side = LEFT, fill = "both", expand = "yes")

		self.master.update_idletasks()


	def plot_network(self, frame, Aij):

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
					frame.canvas.create_line(node_coord[n][1]+40, node_coord[n][0]+20,
								       node_coord[m][1]+40, node_coord[m][0]+20,
										fill="red", width=3)

	def write_run(self):

		current_dir = os.getcwd()
		fig_dir = current_dir + '/fig/'
		data_dir = current_dir + '/data/'

		for input_file_name in self.input_files:
			image_name = input_file_name.split('/')[-1]
			fig_name = ut.check_file_name(image_name, extension='tif')

			image = ImageTk.PhotoImage(Image.open(input_file_name))

			self.display_image_box(self.image_display, image) 

			averages = analyse_image(current_dir, input_file_name, ow_anis=self.ow_anis.get(), 
					ow_graph=self.ow_graph.get(), sigma=0.5, mode='test', noise_thresh=0.18)

			Aij = nx.read_gpickle(data_dir + fig_name + ".pkl")

			self.plot_network(self.image_display, Aij)

		print("Analysis Ended")

side = LEFT
anchor = W

root = Tk()
GUI = imagecol_gui(root)

root.mainloop()
