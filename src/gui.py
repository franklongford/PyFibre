import os, sys
from tkinter import *
from tkinter import ttk

from main import *
import utilities as ut

# if you are still working under a Python 2 version, 
# comment out the previous line and uncomment the following line
# import Tkinter as tk

def write_run():

	modules = ['anisotropy']

	ut.logo()

	run_imagecol(current_dir, input_files, modules, ow_anis.get(), ow_graph.get())

	sys.exit()

side = LEFT
anchor = W

root = Tk()
root.title("ImageCol - Fiborours Tissue Image Toolkit")

frame = Frame(root)
frame.pack()

ow_anis = IntVar()
chk = Checkbutton(frame, text="overwrite anisotropy", variable=ow_anis)
#chk.pack(side=side, anchor=anchor, expand=YES)

ow_graph = IntVar()
chk = Checkbutton(frame, text="overwrite graph", variable=ow_graph)
chk.grid(column=1, row=0, sticky=(N,W,E,S))
#chk.pack(side=side, anchor=anchor, expand=YES)

run_button = Button(frame, width=25,
                   text="RUN",
                   command=write_run)
run_button.grid(column=0, row=2)

current_dir = os.getcwd()
input_files = os.listdir(current_dir)

removed_files = []
for file_name in input_files:
	if not (file_name.endswith('.tif')): removed_files.append(file_name)
	elif (file_name.find('display') != -1): removed_files.append(file_name)
	elif (file_name.find('AVG') == -1): removed_files.append(file_name)
for file_name in removed_files: input_files.remove(file_name)

frame = Frame(root)
frame.pack()

Lb1 = Listbox(frame)
Lb1.grid(column=0, row=1, sticky=(N,W,E,S))

s = ttk.Scrollbar(frame, orient=VERTICAL, command=Lb1.yview)
s.grid(column=1, row=1, sticky=(N,S))
Lb1['yscrollcommand'] = s.set

ttk.Sizegrip(frame).grid(column=1, row=1, sticky=(S,E))

root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(1, weight=1)

for i, filename in enumerate(input_files):
	Lb1.insert(i, filename)

root.mainloop()