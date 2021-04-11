from envisage.core_plugin import CorePlugin
from envisage.ui.tasks.tasks_plugin import TasksPlugin

from pyfibre.gui.image_tab import ImageTab, NetworkImageTab
from pyfibre.gui.metric_tab import ImageMetricTab
from pyfibre.gui.segment_image_tab import SegmentImageTab
from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.gui.file_display_pane import TableRow


from .multi_images import ProbeMultiImage
from .parsers import ProbeFileSet
from .plugins import ProbePyFibreGUIPlugin
from .objects import ProbeFibreNetwork, ProbeSegment


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
        kwargs['tag'] = 'Probe'
        kwargs['file_set'] = ProbeFileSet()
        super().__init__(*args, **kwargs)


class ProbeImageMetricTab(ImageMetricTab):

    def __init__(self, *args, **kwargs):
        kwargs['label'] = 'Test Image'
        super().__init__(*args, **kwargs)
        self.multi_image = ProbeMultiImage()