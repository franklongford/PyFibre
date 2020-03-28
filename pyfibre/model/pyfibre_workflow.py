import copy

from traits.api import HasStrictTraits, Bool, Float, Tuple, Int

from pyfibre.io.utilities import pop_under_recursive


class PyFibreWorkflow(HasStrictTraits):

    #: Unit of scale to resize image
    scale = Float(1.25)

    #: Parameters for non-linear means denoise algorithm
    #: (used to remove noise)
    p_denoise = Tuple((Int, Int))

    #: Standard deviation of Gaussian smoothing
    sigma = Float(0.5)

    #: Metric for hysterisis segmentation
    alpha = Float(0.5)

    #: Toggles force overwrite of existing fibre network
    ow_network = Bool(False)

    #: Toggles force overwrite of existing segmentation
    ow_segment = Bool(False)

    #: Toggles force overwrite of existing metric analysis
    ow_metric = Bool(False)

    #: Toggles creation of figures
    save_figures = Bool(False)

    def _p_denoise_default(self):
        return (5, 35)

    def __getstate__(self):
        state = pop_under_recursive(copy.copy(self.__dict__))
        return state
