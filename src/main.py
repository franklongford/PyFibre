"""
ImageCol: Collagen Image Analysis Program
MAIN ROUTINE 

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 16/08/2018
"""

import sys, os
import utilities as ut

current_dir = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))

ut.logo()

input_files = os.listdir(current_dir)

modules = []

if ('learning' in sys.argv): modules.append('learning')
if ('anisotropy' in sys.argv): modules.append('anisotropy')

if ('learning' in modules):
	from machine_learning import learning
	learning(current_dir)

if ('anisotropy' in modules):
	from anisotropy import analyse_image, analyse_directory
	if ('-all' in sys.argv): analyse_directory(current_dir, input_files)	
	if ('-key' in sys.argv):
		key = sys.argv[sys.argv.index('-key') + 1]
		analyse_directory(current_dir, input_files, key=key)
	elif ('-name' in sys.argv): 
		input_file_name = sys.argv[sys.argv.index('-name') + 1]
		analyse_image(current_dir, input_file_name)
		