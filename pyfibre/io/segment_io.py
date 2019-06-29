import pickle


def save_segment(regions, file_name, file_type=''):
    "Saves scikit image regions as pickled file"

    try:
        with open(f"{file_name}_{file_type}.pkl", 'wb') as outfile:
            pickle.dump(regions, outfile, pickle.HIGHEST_PROTOCOL)
    except IOError as e:
        raise IOError(f"Cannot save to file {file_name}") from e


def load_segment(file_name, file_type=''):
    "Loads pickled scikit image regions"

    try:
        with open(f"{file_name}_{file_type}.pkl", 'rb') as infile:
            regions = pickle.load(infile)
        return regions
    except IOError as e:
        raise IOError(f"Cannot read file {file_name}_{file_type}.pkl") from e
