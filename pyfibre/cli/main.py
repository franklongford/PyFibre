"""
PyFibre: Fiborous Image Analysis Program
MAIN ROUTINE 

Created by: Frank Longford
Created on: 16/08/2018

Last Modified: 18/02/2019
"""

import os
import argparse
import logging

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import pyfibre.utilities as ut
from pyfibre.pyfibre_analyse_image import analyse_image

logger = logging.getLogger(__name__)


def pyfibre_cli():

	current_dir = os.getcwd()
	dir_path = os.path.dirname(os.path.realpath(__file__))

	print(ut.logo())

	parser = argparse.ArgumentParser(description='Image analysis of fibourous tissue samples')
	parser.add_argument('--name', nargs='?', help='Tif file names to load', default="")
	parser.add_argument('--dir', nargs='?', help='Directories to load tif files', default="")
	parser.add_argument('--key', nargs='?', help='Keywords to filter file names', default="")
	parser.add_argument('--sigma', type=float, nargs='?', help='Gaussian smoothing standard deviation', default=0.5)
	parser.add_argument('--alpha', type=float, nargs='?', help='Alpha network coefficient', default=0.5)
	parser.add_argument('--ow_metric', action='store_true', help='Toggles overwrite analytic metrics')
	parser.add_argument('--ow_segment', action='store_true', help='Toggles overwrite image segmentation')
	parser.add_argument('--ow_network', action='store_true', help='Toggles overwrite network extraction')
	parser.add_argument('--ow_figure', action='store_true', help='Toggles overwrite figures')
	parser.add_argument('--save_db', nargs='?', help='Output database filename', default=None)
	parser.add_argument('--threads', type=int, nargs='?', help='Number of threads per processor', default=8)
	parser.add_argument('--debug', action='store_true', help='Toggles debug mode')
	args = parser.parse_args()

	if args.debug:
		logger = ut.setup_logger('DEBUG')
	else:
		logger = ut.setup_logger('INFO')

	input_files = []

	for file_name in args.name.split(','):
		if (file_name.find('/') == -1): file_name = current_dir + '/' + file_name
		input_files.append(file_name)

	if len(args.dir) != 0:
		for directory in args.dir.split(','): 
			for file_name in os.listdir(directory): input_files += [directory + '/' + file_name]

	removed_files = []

	for key in args.key.split(','):
		for file_name in input_files:
			if (file_name.find(key) == -1) and (file_name not in removed_files): 
				removed_files.append(file_name)
		
	for file_name in removed_files: input_files.remove(file_name)

	files, prefixes = ut.get_image_lists(input_files)

	cell_database = pd.DataFrame()
	segment_database = pd.DataFrame()
	global_database = pd.DataFrame()
   
	for i, input_file_names in enumerate(files):

		image_path = '/'.join(prefixes[i].split('/')[:-1])

		data_global, data_segment, data_cell = analyse_image(input_file_names, 
			prefixes[i], image_path, sigma=args.sigma, 
			ow_metric=args.ow_metric, ow_segment=args.ow_segment, 
			ow_network=args.ow_network, ow_figure=args.ow_figure, 
			threads=args.threads, alpha=args.alpha)

		global_database = pd.concat([global_database, data_global])
		segment_database = pd.concat([segment_database, data_segment])
		cell_database = pd.concat([cell_database, data_cell])

		logger.debug(image_path)
		logger.debug("Global Image Analysis Metrics:")
		logger.debug(data_global.iloc[0])


	if args.save_db != None: 
		global_database.to_pickle('{}.pkl'.format(args.save_db))
		global_database.to_excel('{}.xls'.format(args.save_db))
		
		segment_database.to_pickle('{}_fibre.pkl'.format(args.save_db))
		segment_database.to_excel('{}_fibre.xls'.format(args.save_db))

		cell_database.to_pickle('{}_cell.pkl'.format(args.save_db))
		cell_database.to_excel('{}_cell.xls'.format(args.save_db))


if __name__ == '__main__':
	pyfibre_cli()
