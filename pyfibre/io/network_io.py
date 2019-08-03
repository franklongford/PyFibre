import networkx as nx

from pyfibre.utilities import save_pickle, load_pickle


def save_network(network, file_name, file_type=None):
    """Saves pickled networkx graphs"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        nx.write_gpickle(network, f"{file_name}.pkl")
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}.pkl"
        ) from e


def load_network(file_name, file_type=None):
    """Loads pickled networkx graphs"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        network = nx.read_gpickle(f"{file_name}.pkl")
        return network
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}.pkl"
        ) from e


def save_network_list(network_list, file_name, file_type=None):
    """Loads pickled networkx graphs"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        save_pickle(network_list, f"{file_name}.pkl")
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}.pkl"
        ) from e


def load_network_list(file_name, file_type=None):
    """Loads pickled networkx graphs"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        network = load_pickle(f"{file_name}.pkl")
        return network
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}.pkl"
        ) from e