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

if ('-all' in sys.argv): 
	input_files = os.listdir(current_dir)
if ('-key' in sys.argv):
	input_files = os.listdir(current_dir)
	key = sys.argv[sys.argv.index('-key') + 1]
elif ('-name' in sys.argv): 
	input_file_name = sys.argv[sys.argv.index('-name') + 1]

modules = []

if ('learning' in sys.argv): modules.append('learning')
if ('experimental' in sys.argv): modules.append('experimental')

if ('learning' in modules):
	from machine_learning import learning
	learning(current_dir)

if ('experimental' in modules):
	from analysis import experimental_directory
	experimental_directory(current_dir, input_files, key=key)