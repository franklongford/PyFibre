from chaco.api import ArrayPlotData
from enable.component_editor import ComponentEditor
from traits.api import (
    List, Instance, Either, Str, on_trait_change, Tuple, Any,
    Property)
from traitsui.api import (
    TabularEditor, View, UItem, VGroup, EnumEditor, HGroup, Item)
from traitsui.tabular_adapter import TabularAdapter

from pyfibre.gui.image_tab import ImageTab


class ImageMetricTab(ImageTab):

    data = Any

    _data = Property(List(Tuple), depends_on='data')

    headers = Property(List(Str), depends_on='data')

    tabular_adapter = Instance(TabularAdapter, ())

    x_label = Str

    y_label = Str

    _display_cols = Property(List(Str), depends_on='data')

    #: Selected evaluation steps in the table
    _selected_rows = Either(List(Tuple), None)

    def default_traits_view(self):
        editor = TabularEditor(
            adapter=self.tabular_adapter,
            show_titles=True,
            selected="_selected_rows",
            auto_update=False,
            multi_select=True,
            editable=False,
        )

        return View(
            VGroup(
                HGroup(
                    VGroup(
                        UItem('selected_label',
                              editor=EnumEditor(
                                  name='object.image_labels'),
                              style='simple'),
                        UItem('image_plot',
                              editor=ComponentEditor(),
                              show_label=False),
                    ),
                    VGroup(
                        HGroup(
                            Item("x_label",
                                 editor=EnumEditor(name="_display_cols")),
                            Item("y_label",
                                 editor=EnumEditor(name="_display_cols")),
                        ),
                        UItem('component',
                              editor=ComponentEditor(),
                              show_label=False),
                    ),
                ),
                UItem("_data", editor=editor),
                layout="split"
            )
        )

    def _plot_data_default(self):
        plot_data = ArrayPlotData()
        for data in ['x', 'y']:
            plot_data.set_data(data, [])
        return plot_data

    def _get__data(self):
        if self.data is None:
            return []
        fibre_data = self.data.to_records()
        return fibre_data.tolist()

    def _get__display_cols(self):
        if self.data is None:
            return []
        display_cols = [
            name for dtype, name in zip(
                self.data.dtypes, self.data.columns)
            if dtype in [int, float]
        ]
        print(display_cols)
        return display_cols

    def _get_headers(self):
        if self.data is None:
            return []
        return [''] + list(self.data.columns)

    def customise_plot(self, plot):
        plot.plot(("x", "y"), type="scatter", color="blue")

    def _tabular_adapter_default(self):
        return TabularAdapter(columns=self.headers)

    @on_trait_change("headers")
    def _update_adapter(self):
        self.tabular_adapter.columns = self.headers

    @on_trait_change("x_label")
    def _update_plot_x_data(self):
        """ Update data points displayed by the x axis.
        This method is called when the `x` axis is changed.
        """
        if self.x_label == "":
            self.plot_data.set_data("x", [])
        else:
            self.plot.x_axis.title = self.x_label
            index = self.headers.index(self.x_label)
            x_data = [row[index] for row in self._data]
            self.plot_data.set_data("x", x_data)

    @on_trait_change("y_label")
    def _update_plot_y_data(self):
        """ Update data points displayed by the y axis.
        This method is called when the `y` axis is changed.
        """
        if self.y_label == "":
            self.plot_data.set_data("y", [])
        else:
            self.plot.y_axis.title = self.y_label
            index = self.headers.index(self.y_label)
            y_data = [row[index] for row in self._data]
            self.plot_data.set_data("y", y_data)
