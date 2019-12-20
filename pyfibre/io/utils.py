import os


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


def pop_recursive(dictionary, remove_key):
    """Recursively remove a named key from dictionary and any contained
    dictionaries."""
    try:
        dictionary.pop(remove_key)
    except KeyError:
        pass

    for key, value in dictionary.items():
        # If remove_key is in the dict, remove it
        if isinstance(value, dict):
            pop_recursive(value, remove_key)
        # If we have a non-dict iterable which contains a dict,
        # call pop.(remove_key) from that as well
        elif isinstance(value, (tuple, list)):
            for element in value:
                if isinstance(element, dict):
                    pop_recursive(element, remove_key)

    return dictionary


def pop_dunder_recursive(dictionary):
    """ Recursively removes all dunder keys from a nested dictionary. """
    keys = [key for key in dictionary.keys()]
    for key in keys:
        if key.startswith('__') and key.endswith('__'):
            dictionary.pop(key)

    for key, value in dictionary.items():
        # Check subdicts for dunder keys
        if isinstance(value, dict):
            pop_dunder_recursive(value)
        # If we have a non-dict iterable which contains a dict,
        # remove dunder keys from that too
        elif isinstance(value, (tuple, list)):
            for element in value:
                if isinstance(element, dict):
                    pop_dunder_recursive(element)

    return dictionary

