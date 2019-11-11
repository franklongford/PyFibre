import networkx as nx
import numpy as np

from pyfibre.model.objects.fibre import Fibre
from pyfibre.model.objects.fibre_network import FibreNetwork


def generate_image():

    image = np.zeros((10, 10))
    image[0:6, 4] += 2
    image[2, 4:8] += 5
    image[8, 1:4] += 10

    labels = np.zeros((10, 10), dtype=int)
    labels[0:6, 4] = 1
    labels[2, 4:8] = 1
    labels[8, 1:4] += 2

    binary = np.zeros((10, 10), dtype=int)
    binary[0:6, 4] = 1
    binary[2, 4:8] = 1
    binary[8, 1:4] = 1

    return image, labels, binary


def generate_probe_graph():

    graph = nx.Graph()
    graph.add_nodes_from([2, 3, 4, 5])
    graph.add_edges_from([[3, 2], [3, 4], [4, 5]])

    graph.nodes[2]['xy'] = np.array([0, 0])
    graph.nodes[3]['xy'] = np.array([1, 1])
    graph.nodes[4]['xy'] = np.array([2, 2])
    graph.nodes[5]['xy'] = np.array([2, 3])

    graph.edges[3, 4]['r'] = np.sqrt(2)
    graph.edges[2, 3]['r'] = np.sqrt(2)
    graph.edges[5, 4]['r'] = 1

    return graph


class ProbeFibre(Fibre):

    def __init__(self, *args, **kwargs):
        super(ProbeFibre, self).__init__(
            graph=generate_probe_graph())


class ProbeFibreNetwork(FibreNetwork):

    def __init__(self, *args, **kwargs):
        super(ProbeFibreNetwork, self).__init__(
            graph=generate_probe_graph())
