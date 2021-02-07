from traits.api import (
    List, Instance, Either, Str, on_trait_change, Tuple)
from traitsui.api import TabularEditor, View, UItem
from traitsui.tabular_adapter import TabularAdapter

from pyfibre.core.base_multi_image_viewer import BaseDisplayTab


class MetricTab(BaseDisplayTab):

    data = List(Tuple)

    headers = List(Str)

    tabular_adapter = Instance(TabularAdapter, ())

    #: Selected evaluation steps in the table
    _selected_rows = Either(List(Tuple), None)

    def customise_plot(self, plot):
        pass

    def _tabular_adapter_default(self):
        return TabularAdapter(columns=self.headers)

    def default_traits_view(self):
        editor = TabularEditor(
            adapter=self.tabular_adapter,
            show_titles=True,
            selected="_selected_rows",
            auto_update=False,
            multi_select=True,
            editable=False,
        )

        return View(UItem("data", editor=editor))

    @on_trait_change("headers")
    def _update_adapter(self):
        self.tabular_adapter.columns = self.headers
