import os

import numpy as np
from networkx import node_link_graph


def parse_files(name=None, directory=None, key=None):

    input_files = []

    if key == '':
        key = None

    if name is not None:
        for file_name in name.split(','):
            if file_name.find('/') == -1:
                file_name = os.getcwd() + '/' + file_name
            input_files.append(file_name)

    if directory is not None:
        for folder in directory.split(','):
            for file_name in os.listdir(folder):
                input_files += [folder + '/' + file_name]

    if key is not None:
        removed_files = []
        for file_name in input_files:
            if ((file_name.find(key) == -1) and
                    (file_name not in removed_files)):
                removed_files.append(file_name)

        for file_name in removed_files:
            input_files.remove(file_name)

    return input_files


def parse_file_path(file_path):

    file_name = None
    directory = None

    if os.path.isfile(file_path):
        file_name = file_path
    elif os.path.isdir(file_path):
        directory = file_path

    return file_name, directory


def remove_under(dictionary):
    keys = [key for key in dictionary.keys()]
    for key in keys:
        if key.startswith('_'):
            dictionary.pop(key)


def remove_dunder(dictionary):

    keys = [key for key in dictionary.keys()]
    for key in keys:
        if key.startswith('__') and key.endswith('__'):
            dictionary.pop(key)


def pop_recursive(dictionary, pop_func):
    """Recursively remove a named key from dictionary and any contained
    dictionaries."""

    pop_func(dictionary)

    for key, value in dictionary.items():
        # If remove_key is in the dict, remove it
        if isinstance(value, dict):
            pop_recursive(value, pop_func)
        # If we have a non-dict iterable which contains a dict,
        # call pop.(remove_key) from that as well
        elif isinstance(value, (tuple, list)):
            for element in value:
                if isinstance(element, dict):
                    pop_recursive(element, pop_func)

    return dictionary


def pop_under_recursive(dictionary):
    """Recursively removes all under keys from a nested dictionary. """

    pop_recursive(dictionary, remove_under)

    return dictionary


def pop_dunder_recursive(dictionary):
    """ Recursively removes all dunder keys from a nested dictionary. """

    pop_recursive(dictionary, remove_dunder)

    return dictionary


def get_networkx_graph(data):
    """Transform JSON serialised data into a
    networkx Graph object"""

    for coord in data['nodes']:
        coord['xy'] = np.asarray(coord['xy'])
        if 'direction' in coord:
            coord['direction'] = np.asarray(coord['direction'])

    return node_link_graph(data)