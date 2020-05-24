import logging

from traits.api import (
    HasStrictTraits, Bool, Float, Tuple)

from pyfibre.io.core.base_multi_image_reader import WrongFileTypeError

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

    #: Toggles force overwrite of existing fibre network
    ow_network = Bool(False)

    #: Toggles force overwrite of existing segmentation
    ow_segment = Bool(False)

    #: Toggles force overwrite of existing metric analysis
    ow_metric = Bool(False)

    #: Toggles creation of figures
    save_figures = Bool(False)

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

    for image_type, inner_dict in dictionary.items():
        for prefix, filenames in inner_dict.items():

            try:
                multi_image = readers[image_type].load_multi_image(
                    filenames, prefix)
            except (KeyError, ImportError, WrongFileTypeError):
                logger.info(f'Cannot read image data for {filenames}')
                continue

            try:
                analyser = analysers[image_type]
            except KeyError:
                logger.info(f'Cannot analyse image data for {filenames}')
                continue

            logger.info(f"Processing image data for {filenames}")

            analyser.multi_image = multi_image
            databases = runner.run_analysis(analyser)

            yield databases
