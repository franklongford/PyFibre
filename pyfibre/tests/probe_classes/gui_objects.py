from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin

from pyfibre.gui.image_tab import ImageTab, NetworkImageTab
from pyfibre.gui.segment_image_tab import SegmentImageTab
from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.gui.file_display_pane import TableRow
from pyfibre.tests.probe_classes.multi_images import ProbeMultiImage
from pyfibre.tests.probe_classes.plugins import ProbePyFibreGUIPlugin
from pyfibre.tests.probe_classes.objects import ProbeFibreNetwork, ProbeSegment

from pyfibre.tests.fixtures import test_shg_pl_trans_image_path


class ProbePyFibreGUI(PyFibreGUI):

    def __init__(self):

        plugins = [CorePlugin(), TasksPlugin(),
                   ProbePyFibreGUIPlugin()]

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


class ProbeTableRow(TableRow):

    def __init__(self, *args, **kwargs):
        kwargs['_dictionary'] = {
            'SHG-PL-Trans': test_shg_pl_trans_image_path}
        super().__init__(*args, **kwargs)
