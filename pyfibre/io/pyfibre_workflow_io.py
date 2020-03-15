from pyfibre.io.utilities import load_json

from pyfibre.model.pyfibre_workflow import PyFibreWorkflow


def load_pyfibre_workflow(filepath):

    data = load_json(filepath)

    return PyFibreWorkflow(**data)
