"""
ImageCol: Collagen Image Analysis Program
MAIN ROUTINE 

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 16/08/2018
"""

import sys, os
import utilities as ut
import argparse
from anisotropy import analyse_image, analyse_directory

def get_args():

	modules = []

	if ('learning' in sys.argv): modules.append('learning')
	if ('anisotropy' in sys.argv): modules.append('anisotropy')

	return modules

def run_imagecol(input_files, modules, ow_anis=False, ow_graph=False):

	analyse_directory(input_files, ow_anis=ow_anis, ow_graph=ow_graph)
		

if __name__ == '__main__':

	current_dir = os.getcwd()
	dir_path = os.path.dirname(os.path.realpath(__file__))

	print(ut.logo())

	parser = argparse.ArgumentParser(description='Image analysis of fibourous tissue samples')
	parser.add_argument('--ow_anis', action='store_true', help='Toggles overwrite anisotropy analysis')
	parser.add_argument('--ow_graph', action='store_true', help='Toggles overwrite graph analysis')
	args = parser.parse_args()

	input_files = os.listdir(current_dir)

	modules = get_args()

	removed_files = []

	for file_name in input_files:
		if not (file_name.endswith('.tif')): removed_files.append(file_name)
		elif (file_name.find('display') != -1): removed_files.append(file_name)
		elif (file_name.find('AVG') == -1): removed_files.append(file_name)

	if ('-key' in sys.argv):
		key = sys.argv[sys.argv.index('-key') + 1]

		for file_name in input_files:
			if (file_name.find(key) == -1) and (file_name not in removed_files): 
				removed_files.append(file_name)

	elif ('-name' in sys.argv): 
		input_file_name = sys.argv[sys.argv.index('-name') + 1]
		input_files = [input_file_name]
		
	for file_name in removed_files: input_files.remove(file_name)

	analyse_directory(input_files, args.ow_anis, args.ow_graph)

		
