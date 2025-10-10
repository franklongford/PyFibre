import networkx as nx


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
