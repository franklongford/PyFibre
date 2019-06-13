import pickle


class SegmentWriter():

    def __init__(self):
        pass

    def save_region(self, regions, file_name):
        "Saves scikit image regions as pickled file"

        with open('{}.pkl'.format(file_name), 'wb') as outfile:
            pickle.dump(regions, outfile, pickle.HIGHEST_PROTOCOL)


class SegmentReader():

    def __init__(self):
        pass

    def load_region(self, file_name):
        "Loads pickled scikit image regions"

        with open('{}.pkl'.format(file_name), 'rb') as infile:
            regions = pickle.load(infile)

        return regions