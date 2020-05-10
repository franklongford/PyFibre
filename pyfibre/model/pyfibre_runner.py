import logging

from pyfibre.model.pyfibre_workflow import PyFibreWorkflow


logger = logging.getLogger(__name__)


class PyFibreRunner:

    def __init__(self, workflow=None):
        """ Set parameters for ImageAnalyser routines

        Parameters
        ----------
        workflow: PyFibreWorkflow
            Instance containing information regarding PyFibre's
            Workflow
        """

        if workflow is not None:
            self.workflow = workflow
        else:
            self.workflow = PyFibreWorkflow()

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
            self.workflow
        )

        logger.debug(f"Analysis options:\n "
                     f"Extract Network = {network}\n "
                     f"Segment Image = {segment}\n "
                     f"Generate Metrics = {metric}\n "
                     f"Save Figures = {self.workflow.save_figures}")

        databases = analyser.image_analysis(self.workflow)

        return databases
