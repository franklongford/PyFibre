from pyfibre.io.utilities import load_json

from pyfibre.model.pyfibre_runner import PyFibreRunner


def load_pyfibre_workflow(filepath):

    data = load_json(filepath)

    return PyFibreRunner(**data)
