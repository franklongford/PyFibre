from pyfibre.model.tools.analysis import network_analysis, segment_analysis
from pyfibre.model.tools.fibre_assignment import FibreAssignment
from pyfibre.model.tools.fibre_utilities import simplify_network

from .base_graph_segment import BaseGraphSegment


class FibreNetwork(BaseGraphSegment):
    """Container for a Networkx Graph and scikit-image region
    representing a connected fibrous region"""

    def __init__(self, *args, fibres=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.fibres = fibres

        self._area_threshold = 200
        self._iterations = 5

    def __getstate__(self):
        """Return the object state in a form that can be
        serialised as a JSON file"""
        state = super().__getstate__()
        state.pop('fibres', None)
        return state

    @property
    def fibre_assigner(self):
        return FibreAssignment(image=self.image)

    @property
    def red_graph(self):
        return simplify_network(self.graph)

    def generate_fibres(self):
        return self.fibre_assigner.assign_fibres(self.graph)

    def generate_database(self, image=None):
        """Generates a Pandas database with all graph and segment metrics
        for assigned image"""

        database = network_analysis(self.graph, self.red_graph, 'SHG')

        if image is not None:
            segment_metrics = segment_analysis(
                self.segment, image=image, tag='Network')

        else:
            segment_metrics = segment_analysis(
                self.segment, tag='Network')

        database = database.append(segment_metrics, ignore_index=False)

        return database
