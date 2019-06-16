import pickle


def save_segment(regions, file_name):
    "Saves scikit image regions as pickled file"

    try:
        with open('{}.pkl'.format(file_name), 'wb') as outfile:
            pickle.dump(regions, outfile, pickle.HIGHEST_PROTOCOL)
    except IOError as e:
        raise IOError(f"Cannot save to file {file_name}") from e


def load_segment(file_name):
    "Loads pickled scikit image regions"

    try:
        with open('{}.pkl'.format(file_name), 'rb') as infile:
            regions = pickle.load(infile)
        return regions
    except IOError as e:
        raise IOError(f"Cannot read file {file_name}") from e