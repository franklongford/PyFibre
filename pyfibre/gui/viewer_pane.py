from PIL import ImageTk, Image

from pyface.tasks.api import TraitsTaskPane
from pyface.api import ImageResource

from traits.api import (
    HasTraits, Instance, Unicode, List
)
from traitsui.api import (
    View, VGroup, Group, UItem, ImageEditor, HGroup,
    Spring, Image, Item, ListEditor
)


class ImageTab(HasTraits):

    name = Instance(Unicode)

    image = Instance(Image)

    traits_view = View(
        Item('image',
             editor=ImageEditor()
             )
    )


class MetricTab(HasTraits):
    pass


class ViewerPane(TraitsTaskPane):

    id = 'pyfibre.viewer_pane'

    name = 'Viewer Pane'

    image_tab_list = List(ImageTab)

    shg_image_tab = Instance(ImageTab)

    pl_image_tab = Instance(ImageTab)

    tran_image_tab = Instance(ImageTab)

    tensor_tab = Instance(ImageTab)

    network_tab = Instance(ImageTab)

    segment_tab = Instance(ImageTab)

    fibre_tab = Instance(ImageTab)

    cell_tab = Instance(ImageTab)

    metric_tab = Instance(MetricTab)

    list_editor = ListEditor()

    traits_view = View(
        Group(
            Item('image_tab_list',
                 editor=list_editor),
        )
    )

    def _image_tab_list_default(self):

        return [self.shg_image_tab,
                self.pl_image_tab]

