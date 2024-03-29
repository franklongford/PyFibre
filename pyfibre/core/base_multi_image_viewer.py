from abc import abstractmethod

from chaco.api import AbstractPlotData, Plot
from enable.api import ComponentEditor, Component
from traits.api import (
    ABCHasStrictTraits, Instance, Property, List, Str, provides)
from traitsui.api import Item, View, ListEditor, Group

from .base_multi_image import BaseMultiImage
from .i_multi_image_viewer import IMultiImageViewer, IDisplayTab


@provides(IDisplayTab)
class BaseDisplayTab(ABCHasStrictTraits):

    # Label of image tab
    label = Str()

    #: Reference to the Chaco component to be displayed in the TraitsUI view
    component = Instance(Component)

    # Plot object to display in tab
    plot = Property(Instance(Plot), depends_on="plot_data")

    # Data to be displayed in Plot object
    plot_data = Instance(AbstractPlotData)

    # Simple view displaying plot attribute
    trait_view = View(
            Item('component',
                 editor=ComponentEditor(),
                 show_label=False),
            resizable=True
        )

    def _component_default(self):
        return self.plot

    def _get_plot(self):
        """Returns chaco Plot object with formatting provided
        by plot_data attribute.
        """

        # Generate Plot object
        plot = Plot(self.plot_data)

        # Apply optional tools and formatting
        self.customise_plot(plot)

        return plot

    @abstractmethod
    def update_tab(self):
        """Provide additional instructions to update other components
        of the tab.
        """

    @abstractmethod
    def customise_plot(self, plot):
        """Provide additional customisation to chaco Plot object
        generated by this class.
        """


@provides(IMultiImageViewer)
class BaseMultiImageViewer(ABCHasStrictTraits):

    multi_image = Instance(BaseMultiImage)

    display_tabs = List(IDisplayTab)

    selected_tab = Instance(IDisplayTab)

    def default_traits_view(self):

        list_editor = ListEditor(
            page_name='.label',
            use_notebook=True,
            dock_style='tab',
            style='custom',
            selected='object.selected_tab'
        )

        traits_view = View(
            Group(
                Item('display_tabs',
                     editor=list_editor,
                     style='custom'),
                show_labels=False
            )
        )

        return traits_view

    def _display_tabs_default(self):
        return self.create_display_tabs()

    def _selected_tab_default(self):
        if self.display_tabs:
            return self.display_tabs[0]

    def update_viewer(self, multi_image):
        """Sets the multi_image attribute and updates all display tabs"""
        self.multi_image = multi_image
        self.update_display_tabs()

    @abstractmethod
    def create_display_tabs(self):
        """Returns a list of objects providing the IDisplayTab
        interface
        """

    @abstractmethod
    def update_display_tabs(self):
        """Updates each display tab when called"""
