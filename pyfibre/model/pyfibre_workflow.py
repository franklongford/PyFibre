from traits.api import (
    HasStrictTraits, Bool, Float, Tuple, Int, Dict)


class PyFibreWorkflow(HasStrictTraits):

    #: Unit of scale to resize image
    scale = Float(1.25)

    #: Parameters for non-linear means algorithm
    #: (used to remove noise)
    p_denoise = Tuple((Int, Int))

    #: Standard deviation of Gaussian smoothing
    sigma = Float(0.5)

    #: Metric for hysterisis segmentation
    alpha = Float(0.5)

    #: Parameters used for FIRE algorithm
    fire_parameters = Dict()

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

    def _fire_parameters_default(self):
        return {
            'nuc_thresh': 2,
            'nuc_radius': 11,
            'lmp_thresh': 0.15,
            'angle_thresh': 70,
            'r_thresh': 7
        }
