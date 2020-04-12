import networkx as nx
import numpy as np
from skimage.io import imread
from skimage.measure import regionprops

from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin

from pyfibre.gui.image_tab import ImageTab, NetworkImageTab
from pyfibre.gui.segment_image_tab import SegmentImageTab
from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.gui.pyfibre_main_task import PyFibreMainTask
from pyfibre.gui.file_display_pane import TableRow
from pyfibre.gui.pyfibre_plugin import PyFibrePlugin
from pyfibre.model.multi_image.fixed_stack_image import FixedStackImage
from pyfibre.model.objects.base_graph import BaseGraph
from pyfibre.model.objects.base_segment import BaseSegment
from pyfibre.model.objects.base_graph_segment import BaseGraphSegment
from pyfibre.model.objects.fibre import Fibre
from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.model.multi_image.base_multi_image import BaseMultiImage
from pyfibre.model.multi_image.multi_images import SHGImage, SHGPLTransImage

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


class ProbeGraphMixin:

    def __init__(self, *args, **kwargs):
        kwargs['graph'] = generate_probe_graph()
        super().__init__(
            *args, **kwargs)


class ProbeGraphSegment(BaseGraphSegment):

    def __init__(self, *args, **kwargs):
        kwargs['graph'] = generate_probe_graph()
        kwargs['shape'] = (3, 4)
        super().__init__(
            *args, **kwargs)

    def generate_database(self, image=None):
        pass


class ProbeGraph(ProbeGraphMixin, BaseGraph):

    def generate_database(self, image=None):
        pass


class ProbeFibre(ProbeGraphMixin, Fibre):
    pass


class ProbeFibreNetwork(ProbeGraphMixin, FibreNetwork):
    pass


class ProbeSegment(BaseSegment):

    _tag = 'Test'

    def __init__(self, *args, **kwargs):
        kwargs['region'] = generate_regions()[0]
        super(ProbeSegment, self).__init__(
            *args, **kwargs)


class ProbeMultiImage(BaseMultiImage):

    def __init__(self, *args, **kwargs):
        image, _, _, _ = generate_image()
        kwargs['image_stack'] = [image, 2 * image]
        super().__init__(*args, **kwargs)
        self.image_dict = {
            'Test 1': self.image_stack[0],
            'Test 2': self.image_stack[1]}

    def preprocess_images(self):
        pass

    def verify_stack(cls, image_stack):
        pass

    def segmentation_algorithm(self, *args, **kwargs):
        pass

    def create_figures(self, *args, **kwargs):
        pass


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

        # 'Run' the application by creating windows
        # without an event loop
        self.run = self._create_windows


class ProbeImageTab(ImageTab):

    def __init__(self, *args, **kwargs):
        kwargs['label'] = 'Test Image'
        super().__init__(*args, **kwargs)
        self.multi_image = ProbeMultiImage()


class ProbeNetworkImageTab(NetworkImageTab):

    def __init__(self, *args, **kwargs):
        kwargs['networks'] = [ProbeFibreNetwork().graph]
        super().__init__(*args, **kwargs)
        self.multi_image = ProbeMultiImage()


class ProbeSegmentImageTab(SegmentImageTab):

    def __init__(self, *args, **kwargs):
        kwargs['segments'] = [ProbeSegment()]
        super().__init__(*args, **kwargs)
        self.multi_image = ProbeMultiImage()


class ProbeFixedStackImage(FixedStackImage):

    _stack_len = 1

    _allowed_dim = [2]


class ProbeTableRow(TableRow):

    def __init__(self, *args, **kwargs):
        kwargs['_dictionary'] = {
            'SHG-PL-Trans': test_shg_pl_trans_image_path}
        super().__init__(*args, **kwargs)
