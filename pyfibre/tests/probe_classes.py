import networkx as nx
import numpy as np
from skimage.io import imread
from skimage.measure import regionprops

from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin

from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.gui.pyfibre_main_task import PyFibreMainTask
from pyfibre.gui.pyfibre_plugin import PyFibrePlugin

from pyfibre.model.objects.fibre import Fibre
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.model.objects.multi_image import SHGImage, SHGPLTransImage
from pyfibre.model.objects.segments import FibreSegment

from .fixtures import test_shg_image_path, test_shg_pl_trans_image_path


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


def generate_regions():

    image, labels, _, _ = generate_image()

    regions = regionprops(labels, intensity_image=image)

    return regions


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


class ProbeSHGImage(SHGImage):

    def __init__(self, *args, **kwargs):
        kwargs.pop('image_stack', None)

        image = imread(test_shg_image_path)

        image = np.mean(image, axis=-1)
        image = image / image.max()
        image_stack = [image]

        super(ProbeSHGImage, self).__init__(
            *args, image_stack=image_stack, **kwargs
        )


class ProbeSHGPLTransImage(SHGPLTransImage):

    def __init__(self, *args, **kwargs):
        kwargs.pop('image_stack', None)

        images = imread(test_shg_pl_trans_image_path)

        image_stack = []
        for image in images:
            image = np.mean(image, axis=-1)
            image = image / image.max()
            image_stack.append(image)

        super(ProbeSHGPLTransImage, self).__init__(
            *args, image_stack=image_stack, **kwargs
        )


class ProbePyFibrePlugin(PyFibrePlugin):

    def _create_main_task(self):
        pyfibre_task = PyFibreMainTask()
        return pyfibre_task


class ProbePyFibreGUI(PyFibreGUI):

    def __init__(self):

        plugins = [CorePlugin(), TasksPlugin(),
                   ProbePyFibrePlugin()]

        super(ProbePyFibreGUI, self).__init__(plugins=plugins)

        # 'Run' the application by creating windows without an event loop
        self.run = self._create_windows
