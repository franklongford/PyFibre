from pyfibre.model.tools.convertors import networks_to_segments
from pyfibre.model.tools.fibre_utilities import get_node_coord_array


class BaseGraphSegment:
    """Container for a Networkx Graph and scikit-image segment
    representing a connected fibrous region"""

    def __init__(self, graph=None, image=None):

        self.graph = graph
        self._area_threshold = 64
        self._iterations = 2
        self._sigma = 0.5

        self.image = image

    @property
    def node_list(self):
        return list(self.graph.nodes)

    @property
    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def node_coord(self):
        return get_node_coord_array(self.graph)

    @property
    def segment(self):
        """Scikit-image segment"""
        if self.image is None:
            range_x = self.node_coord[:, 0].max() - self.node_coord[:, 0].min() + 1
            range_y = self.node_coord[:, 1].max() - self.node_coord[:, 1].min() + 1
            shape = (int(range_x), int(range_y))
            segments = networks_to_segments(
                [self.graph], shape=shape,
                area_threshold=self._area_threshold,
                iterations=self._iterations,
                sigma=self._sigma)
        else:
            segments = networks_to_segments(
                [self.graph], image=self.image,
                area_threshold=self._area_threshold,
                iterations=self._iterations,
                sigma=self._sigma)

        return segments[0]

    def add_node(self, *args, **kwargs):
        """Add node to Networkx graph attribute"""
        self.graph.add_node(*args, **kwargs)

    def add_edge(self, *args, **kwargs):
        """Add edge to Networkx graph attribute"""
        self.graph.add_edge(*args, **kwargs)

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""
        raise NotImplementedError()