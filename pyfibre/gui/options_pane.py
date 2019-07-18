from pyface.tasks.api import TraitsDockPane
from pyface.api import ImageResource

from traits.api import (
    Bool, Float, List, Instance
)
from traitsui.api import (
    View, VGroup, Item, InstanceEditor, UItem,
    ImageEditor
)


class OptionsPane(TraitsDockPane):

    id = 'pyfibre.options_pane'

    #ui.visible = Bool(False)

    # Overwrite options
    ow_metric = Bool(False)

    ow_segment = Bool(False)

    ow_network = Bool(False)

    ow_figure = Bool(False)

    # Database options
    save_database = Bool(False)

    # Image analysis parameters
    sigma = Float(0.5)

    p_intensity = List([1, 99])

    p_denoise = List([5, 35])

    alpha = Float(0.5)

    image_editor = ImageEditor(scale=True,
                               allow_upscaling=False,
                               preserve_aspect_ratio=True)

    traits_view = View(
        VGroup(
            Item('ow_metric'),
            Item('ow_segment'),
            Item('ow_network'),
            Item('ow_figure'),
            Item('save_database'),
            Item('sigma'),
            Item('alpha'),
            Item('p_intensity',
                  editor=InstanceEditor(),
                  style='custom'
                 ),
            Item('p_denoise',
                 editor=InstanceEditor(),
                 style='custom'
                 ),
        )
    )
