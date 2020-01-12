from pyface.tasks.api import TraitsDockPane
from pyface.api import ImageResource

from traits.api import (
    Bool, Float, List, Instance, Int, Button, File
)
from traitsui.api import (
    View, VGroup, Item, InstanceEditor, UItem,
    ImageEditor, RangeEditor, Group
)


class OptionsPane(TraitsDockPane):

    id = 'pyfibre.options_pane'

    name = 'Options Pane'

    #: Remove the possibility to close the pane
    closable = False

    #: Remove the possibility to detach the pane from the GUI
    floatable = False

    #: Remove the possibility to move the pane in the GUI
    movable = False

    #: Make the pane visible by default
    visible = True

    #: SHG and PL options
    shg_required = Bool(True)

    pl_required = Bool(True)

    # Overwrite options
    ow_metric = Bool(False)

    ow_segment = Bool(False)

    ow_network = Bool(False)

    save_database = Bool(False)

    database_filename = File('pyfibre_database')

    # Image analysis parameters
    sigma = Float(0.5)

    low_intensity = Int(1)

    high_intensity = Int(99)

    n_denoise = Int(5)

    m_denoise = Int(35)

    alpha = Float(0.5)

    image_editor = ImageEditor(scale=True,
                               allow_upscaling=False,
                               preserve_aspect_ratio=True)

    int_range_editor = RangeEditor(low=1, high=100, mode='slider')

    pix_range_editor = RangeEditor(low=2, high=50, mode='slider')

    traits_view = View(
        VGroup(
            Item('shg_required'),
            Item('pl_required'),
            Item('ow_network', label="Overwrite Network?"),
            Item('ow_segment', label="Overwrite Segments?"),
            Item('ow_metric', label="Overwrite Metrics?"),
            Item('save_database', label='Save Database?'),
            Item('database_filename',
                 label='Database file',
                 visible_when='save_database'),
            Item('sigma', label="Gaussian Std Dev (pix)"),
            Item('alpha', label="Alpha network coefficient"),
            Group(
                Item('low_intensity',
                     editor=int_range_editor,
                     style='custom',
                     label="Low Clip Intensity (%)"
                     ),
                Item('high_intensity',
                     editor=int_range_editor,
                     style='custom',
                     label="High Clip Intensity (%)"
                     )
            ),
            Group(
                Item('n_denoise',
                     editor=pix_range_editor,
                     style='custom',
                     label="NL-Mean Neighbourhood 1 (pix)"
                     ),
                Item('m_denoise',
                     editor=pix_range_editor,
                     style='custom',
                     label="NL-Mean Neighbourhood 2 (pix)"
                     )
            )
        )
    )

