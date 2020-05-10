from pyfibre.model.core.base_graph import BaseGraph
from pyfibre.model.core.base_graph_segment import BaseGraphSegment
from pyfibre.model.core.base_segment import BaseSegment
from pyfibre.model.objects.fibre import Fibre
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.model.objects.segments import FibreSegment, CellSegment
from pyfibre.tests.probe_classes.utilities import (
    generate_probe_graph, generate_regions)


class ProbeGraph(BaseGraph):

    def __init__(self, *args, **kwargs):
        kwargs['graph'] = generate_probe_graph()
        super().__init__(
            *args, **kwargs)

    def generate_database(self, image=None):
        pass


class ProbeGraphSegment(BaseGraphSegment):

    def __init__(self, *args, **kwargs):
        kwargs['graph'] = generate_probe_graph()
        kwargs['shape'] = (3, 4)
        super().__init__(
            *args, **kwargs)

    def generate_database(self, image=None):
        pass


class ProbeFibre(Fibre):

    def __init__(self, *args, **kwargs):
        kwargs['graph'] = generate_probe_graph()
        super().__init__(
            *args, **kwargs)


class ProbeFibreNetwork(FibreNetwork):

    def __init__(self, *args, **kwargs):
        kwargs['graph'] = generate_probe_graph()
        super().__init__(
            *args, **kwargs)
        self.fibres = self.generate_fibres()
        self.red_graph = self.generate_red_graph()


class ProbeSegment(BaseSegment):

    _tag = 'Test'

    def __init__(self, *args, **kwargs):
        if 'region' not in kwargs:
            kwargs['region'] = generate_regions()[0]
        super(ProbeSegment, self).__init__(
            *args, **kwargs)


class ProbeFibreSegment(FibreSegment):

    def __init__(self, *args, **kwargs):
        if 'region' not in kwargs:
            kwargs['region'] = generate_regions()[0]
        super(ProbeFibreSegment, self).__init__(
            *args, **kwargs)


class ProbeCellSegment(CellSegment):

    def __init__(self, *args, **kwargs):
        if 'region' not in kwargs:
            kwargs['region'] = generate_regions()[0]
        super(ProbeCellSegment, self).__init__(
            *args, **kwargs)
