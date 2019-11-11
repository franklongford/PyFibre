from pyfibre.utilities import save_pickle, load_pickle


def save_objects(objects, file_name, file_type=None):
    """Loads pickled image objects"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        save_pickle(objects, f"{file_name}.pkl")
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}.pkl"
        ) from e


def load_objects(file_name, file_type=None):
    """Loads pickled image objects"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    print(file_name)
    try:
        objects = load_pickle(f"{file_name}.pkl")
        return objects
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}.pkl"
        ) from e