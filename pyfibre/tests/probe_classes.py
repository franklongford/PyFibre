import os

import networkx as nx
import numpy as np

from skimage.io import imread

from pyfibre.model.objects.fibre import Fibre
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.model.objects.multi_image import SHGPLTransImage
from pyfibre.tests.fixtures import test_image_path


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

    stack = np.zeros((2, 10, 10))
    stack[0, 0:6, 4] = 1
    stack[0, 2, 4:8] = 1
    stack[1, 8, 1:4] = 1

    return image, labels, binary, stack


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
            graph=generate_probe_graph(),
            shape=(10, 10))


class ProbeFibreNetwork(FibreNetwork):

    def __init__(self, *args, **kwargs):
        super(ProbeFibreNetwork, self).__init__(
            graph=generate_probe_graph(),
            shape=(10, 10))


class ProbeSHGPLTransImage(SHGPLTransImage):

    def __init__(self, *args, **kwargs):
        kwargs.pop('image_stack', None)

        images = imread(test_image_path)

        image_stack = []
        for image in images:
            image = np.mean(image, axis=-1)
            image = image / image.max()
            image_stack.append(image)

        super(ProbeSHGPLTransImage, self).__init__(
            *args, image_stack=image_stack, **kwargs
        )