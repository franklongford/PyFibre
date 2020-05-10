import logging

from traits.api import (
    HasStrictTraits, Bool, Float, Tuple, Dict)

from pyfibre.io.base_multi_image_reader import WrongFileTypeError
from pyfibre.io.shg_pl_reader import assign_images

logger = logging.getLogger(__name__)


class PyFibreRunner(HasStrictTraits):
    """ Set parameters for ImageAnalyser routines """

    #: Unit of scale to resize image
    scale = Float(1.25)

    #: Parameters for non-linear means algorithm
    #: (used to remove noise)
    p_denoise = Tuple((5, 35))

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

    def _fire_parameters_default(self):
        return {
            'nuc_thresh': 2,
            'nuc_radius': 11,
            'lmp_thresh': 0.15,
            'angle_thresh': 70,
            'r_thresh': 7
        }

    def run_analysis(self, analyser):
        """
        Analyse input image by calculating metrics and
        segmenting via FIRE algorithm

        Parameters
        ----------
        analyser: BaseAnalyser
            Contains reference to MultiImage and analysis script
            to be performed

        Returns
        -------
        databases: list of pd.DataFrame
            Calculated metrics for further analysis
        """

        network, segment, metric = analyser.get_analysis_options(
            self
        )

        logger.debug(f"Analysis options:\n "
                     f"Extract Network = {network}\n "
                     f"Segment Image = {segment}\n "
                     f"Generate Metrics = {metric}\n "
                     f"Save Figures = {self.save_figures}")

        databases = analyser.image_analysis(self)

        return databases


def analysis_generator(dictionary, runner, analysers, readers):

    for prefix, data in dictionary.items():

        filenames, image_type = assign_images(data)

        try:
            multi_image = readers[image_type].load_multi_image(
                filenames, prefix)
            analyser = analysers[image_type]

        except (KeyError, ImportError, WrongFileTypeError):
            logger.info(f'Cannot process image data for {filenames}')

        else:
            logger.info(f"Processing image data for {filenames}")

            analyser.multi_image = multi_image
            databases = runner.run_analysis(analyser)

            yield databases