from pyfibre.model.tools.metrics import (
    segment_shape_metrics, network_metrics, segment_texture_metrics)
from pyfibre.model.tools.fibre_assignment import FibreAssignment
from pyfibre.model.tools.fibre_utilities import simplify_network

from .base_graph_segment import BaseGraphSegment
from .fibre import Fibre


class FibreNetwork(BaseGraphSegment):
    """Container for a Networkx Graph and scikit-image region
    representing a connected fibrous region"""

    def __init__(self, *args, fibres=None, **kwargs):
        super().__init__(*args, **kwargs)

        if fibres is None:
            self.fibres = []
        elif isinstance(fibres, list):
            self.fibres = [
                Fibre(image=self.image, **element)
                if isinstance(element, dict)
                else element
                for element in fibres
            ]

        self._area_threshold = 200
        self._iterations = 5

    def __getstate__(self):
        """Return the object state in a form that can be
        serialised as a JSON file"""
        state = super().__getstate__()
        state["fibres"] = [
            fibre.__getstate__()
            for fibre in self.fibres]

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

        database = network_metrics(self.graph, self.red_graph, 'SHG')

        shape_metrics = segment_shape_metrics(
            self.segment, tag='Network')
        texture_metrics = segment_texture_metrics(
            self.segment, image=image, tag='Network')

        database = database.append(shape_metrics, ignore_index=False)
        database = database.append(texture_metrics, ignore_index=False)

        return database
