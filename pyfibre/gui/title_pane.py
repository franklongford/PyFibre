from pyface.tasks.api import TraitsDockPane
from pyface.api import ImageResource

from traitsui.api import (
    View, VGroup, Group, UItem, ImageEditor, HGroup,
    Spring, InstanceEditor
)

def horizontal_centre(item_or_group):
    return HGroup(Spring(), item_or_group, Spring())


class TitlePane(TraitsDockPane):
    # ------------------
    # Regular Attributes
    # ------------------

    #: An internal identifier for this pane
    id = 'pyfibre.title_pane'

    #: Name displayed as the title of this pane
    name = 'Title Pane'

    #: The PyFibre logo. Stored at images/icon.ico
    image = ImageResource('icon.ico')

    def default_traits_view(self):

        traits_view = View(
                Group(
                    UItem('image',
                          editor=ImageEditor(scale=True,
                                             allow_upscaling=False,
                                             preserve_aspect_ratio=True)
                          )
            )
        )

        return traits_view
