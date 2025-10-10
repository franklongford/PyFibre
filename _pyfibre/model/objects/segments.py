from pyfibre.model.core.base_segment import BaseSegment


class CellSegment(BaseSegment):
    """Segment representing a cellular region"""

    def get_tag(self):
        return 'Cell'


class FibreSegment(BaseSegment):
    """Segment representing a fibrous region"""

    def get_tag(self):
        return 'Fibre'
