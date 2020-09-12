import logging

from traits.api import (
    HasStrictTraits, Bool, Float, Tuple)

from pyfibre.core.base_multi_image_reader import WrongFileTypeError

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

    def run(self, file_sets, analyser, reader):
        """Generator that returns databases of metrics from each image
        in dictionary. Analyses input image by calculating metrics and
        segmenting via FIRE algorithm

        Parameters
        ----------
        file_sets: list of IFileSet
            Contains file sets corresponding to each BaseMultiImage to
            be analysed
        analyser: BaseAnalyser
            Contains reference to MultiImage and analysis script
            to be performed
        reader: BaseMultiImageReader
            Contains loading routines for a BaseMultiImage class

        Yields
        ------
        databases: list of pd.DataFrame
            Calculated metrics for further analysis
        """

        for file_set in file_sets:

            try:
                multi_image = reader.load_multi_image(file_set)
            except (ImportError, WrongFileTypeError):
                logger.info(f'Cannot read image data for {file_set}')
                continue

            analyser.multi_image = multi_image

            try:
                logger.info(f"Processing image data for {file_set}")
                databases = analyser.image_analysis(self)
            except Exception:
                logger.info(f'Cannot analyse image data for {file_set}')
                continue

            yield databases
