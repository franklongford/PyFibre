import networkx as nx


def save_network(network, file_name, file_type=''):
    """Saves pickled networkx graphs"""

    try:
        nx.write_gpickle(network, f"{file_name}_{file_type}.pkl")
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}_{file_type}.pkl"
        ) from e


def load_network(file_name, file_type='graph'):
    """Loads pickled networkx graphs"""

    try:
        network = nx.read_gpickle(f"{file_name}_{file_type}.pkl")
        return network
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}_{file_type}.pkl"
        ) from e
