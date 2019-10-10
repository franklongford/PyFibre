import networkx as nx
import numpy as np

from pyfibre.model.tools.fibre_assignment import Fibre


def generate_probe_network():

    graph = nx.Graph()
    graph.add_nodes_from(np.arange(4, dtype=int) + 2)
    graph.add_edges_from([[3, 2], [3, 4], [4, 5]])

    graph.nodes[2]['xy'] = np.array([0, 0])
    graph.nodes[3]['xy'] = np.array([1, 1])
    graph.nodes[4]['xy'] = np.array([2, 2])
    graph.nodes[5]['xy'] = np.array([3, 3.2])

    graph.edges[3, 4]['r'] = np.sqrt(2)
    graph.edges[2, 3]['r'] = np.sqrt(2)
    graph.edges[5, 4]['r'] = np.sqrt(2)

    return graph


class ProbeFibre(Fibre):

    def __init__(self,*args, **kwargs):
        nodes = [0, 1, 2, 3]
        edges = [(0, 1), (1, 2), (2, 3)]

        super(ProbeFibre, self).__init__(
            nodes=nodes, edges=edges, *args, **kwargs
        )

        self.nodes[0]['xy'] = np.array([0, 0])
        self.nodes[1]['xy'] = np.array([1, 1])
        self.nodes[2]['xy'] = np.array([2, 2])
        self.nodes[3]['xy'] = np.array([3, 3.2])

        self.edges[0, 1]['r'] = np.sqrt(2)
        self.edges[1, 2]['r'] = np.sqrt(2)
        self.edges[2, 3]['r'] = np.sqrt(1 + 1.2**2)
