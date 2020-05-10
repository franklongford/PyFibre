from pyfibre.model.tools.convertors import networks_to_regions

from pyfibre.model.core.base_graph import BaseGraph


class BaseGraphSegment(BaseGraph):
    """Container for a Networkx Graph and scikit-image segment
    representing a connected fibrous region"""

    def __init__(self, graph=None, image=None, shape=None):

        super(BaseGraphSegment, self).__init__(graph=graph)

        if image is None and shape is None:
            raise AttributeError(
                'Cannot instantiate BaseGraphSegment class: '
                'either image or shape argument must be declared')

        self.image = image
        self._shape = shape

        self._area_threshold = 64
        self._iterations = 2
        self._sigma = 0.5

    @property
    def shape(self):
        if self.image is not None:
            return self.image.shape
        return self._shape

    @property
    def region(self):
        """Scikit-image segment"""
        if self.image is None:
            regions = networks_to_regions(
                [self.graph], shape=self.shape,
                area_threshold=self._area_threshold,
                iterations=self._iterations, sigma=self._sigma)
        else:
            regions = networks_to_regions(
                [self.graph], image=self.image,
                area_threshold=self._area_threshold,
                iterations=self._iterations, sigma=self._sigma)

        return regions[0]

    def to_json(self):
        """Return the object state in a form that can be
        serialised as a JSON file"""
        state = super(BaseGraphSegment, self).to_json()
        state.pop('image', None)
        state['shape'] = self.shape

        return state
