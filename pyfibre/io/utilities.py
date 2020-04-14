import json
import os

import numpy as np
from networkx import node_link_graph, node_link_data


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


def remove_contraction(dictionary):
    keys = [key for key in dictionary.keys()]
    for key in keys:
        if isinstance(key, str) and key.startswith('contraction'):
            dictionary.pop(key)


def pop_recursive(dictionary, pop_func):
    """Recursively remove a named key from dictionary
    and any contained dictionaries."""

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


def deserialize_networkx_graph(data):
    """Transform JSON serialised data into a
    networkx Graph object"""

    data = python_to_numpy_recursive(data)
    graph = node_link_graph(data)

    return graph


def serialize_networkx_graph(graph):
    """Transform a networkx Graph object into
    a JSON serialised dictionary"""

    data = node_link_data(graph)
    data = numpy_to_python_recursive(data)

    return data


def numpy_to_python_recursive(dictionary):
    """Convert all numpy values in nested dictionary
    to pure python values"""

    for key, value in dictionary.items():

        if isinstance(value, dict):
            numpy_to_python_recursive(value)

        elif isinstance(value, np.ndarray):
            dictionary[key] = value.tolist()

        elif isinstance(value, np.int64):
            dictionary[key] = int(value)

        elif isinstance(value, np.float):
            dictionary[key] = float(value)

        elif isinstance(value, (list, tuple)):
            for element in value:
                if isinstance(element, dict):
                    numpy_to_python_recursive(element)

    return dictionary


def python_to_numpy_recursive(dictionary):
    """Convert all numpy values in nested dictionary
    to pure python values"""

    for key, value in dictionary.items():

        if isinstance(value, dict):
            python_to_numpy_recursive(value)

        elif isinstance(value, list):
            if key in ['xy', 'direction']:
                dictionary[key] = np.array(value)
            else:
                for element in value:
                    if isinstance(element, dict):
                        python_to_numpy_recursive(element)

    return dictionary


def check_string(string, pos, sep, word):
    """Checks index 'pos' of 'string' seperated by 'sep' for substring 'word'
    If present, removes 'word' and returns amended string
    """

    if sep in string:
        temp_string = string.split(sep)
        if temp_string[pos] == word:
            temp_string.pop(pos)
        string = sep.join(temp_string)

    return string


def check_file_name(file_name, file_type="", extension=""):
    """
    check_file_name(file_name, file_type="", extension="")

    Checks file_name for file_type or extension. If present, returns
    amended file_name without extension or file_type

    """

    file_name = check_string(file_name, -1, '.', extension)
    file_name = check_string(file_name, -1, '_', file_type)

    return file_name


def replace_ext(file_name, extension):
    """If an extension exists on file_name,
    replace it with new extension. Otherwise
    add new extension"""

    path, ext = os.path.splitext(file_name)

    if ext != f'.{extension}':
        file_name = path + f'.{extension}'

    return file_name


def get_file_names(prefix):

    image_name = os.path.basename(prefix)
    working_dir = (
        f"{os.path.dirname(prefix)}/{image_name}"
        "-pyfibre-analysis")

    data_dir = working_dir + '/data/'
    fig_dir = working_dir + '/fig/'

    filename = data_dir + image_name
    figname = fig_dir + image_name

    return working_dir, data_dir, fig_dir, filename, figname


def save_json(data, file_name):
    """Saves data as JSON file"""

    file_name = replace_ext(file_name, 'json')

    try:
        with open(f"{file_name}", 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}"
        ) from e


def load_json(file_name):
    """Loads JSON file as data"""

    file_name = replace_ext(file_name, 'json')

    try:
        with open(file_name, 'r') as infile:
            data = json.load(infile)
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}"
        ) from e

    return data


def save_numpy(file_name, array):
    """Saves array as numpy binary"""

    file_name = replace_ext(file_name, 'npy')

    try:
        np.save(file_name, array)
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}"
        ) from e


def load_numpy(file_name):
    """Loads numpy binary file as array"""

    file_name = replace_ext(file_name, 'npy')

    try:
        array = np.load(file_name)
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}"
        ) from e

    return array
