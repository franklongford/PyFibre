import logging

import networkx as nx
import numpy as np
import pandas as pd

from pyfibre.fibers.fibre_network import FibreNetwork
from pyfibre.fibers.fibre import Fibre
from pyfibre.tools.analysis import angle_analysis

logger = logging.getLogger(__name__)

FIBRE_METRICS = ["Waviness", "Length"]
NETWORK_METRICS = ["Degree", "Eigenvalue", "Connectivity", "Cross-Link Density"]


def fibre_metrics(fibres: list[Fibre]) -> pd.DataFrame:
    """Analysis of list of `Fibre` objects

    Parameters
    ----------
    fibres : list of `<class: Fibre>`
        List of fibre to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from networkx Graph
    """

    database = pd.DataFrame()

    for fibre in fibres:
        series = pd.Series(dtype=object)

        series["Fibre Waviness"] = fibre.waviness
        series["Fibre Length"] = fibre.fibre_l
        series["Fibre Angle"] = fibre.angle
        database = pd.concat([database, series.to_frame().T], ignore_index=True)

    return database


def _network_metrics(fibre_network: FibreNetwork) -> pd.Series:
    """Analyse a single FibreNetwork object"""
    series = pd.Series(dtype=object)

    fibres = fibre_network.generate_fibres()
    red_graph = fibre_network.generate_red_graph()
    metrics = fibre_metrics(fibres)

    series["No. Fibres"] = len(fibres)
    series["Fibre Angle SDI"], _ = angle_analysis(metrics["Fibre Angle"].to_numpy())

    metrics = metrics.drop(["Fibre Angle"], axis=1)
    mean_metrics = metrics.mean()

    for metric in FIBRE_METRICS:
        series[f"Mean Fibre {metric}"] = mean_metrics[f"Fibre {metric}"]

    cross_links = np.array(
        [degree[1] for degree in fibre_network.graph.degree], dtype=int
    )
    series["Fibre Network Cross-Link Density"] = (cross_links > 2).sum() / len(fibres)

    try:
        value = (
            nx.degree_pearson_correlation_coefficient(fibre_network.graph, weight="r")
            ** 2
        )
    except Exception as err:
        logger.debug(f"Network Degree calculation failed: {str(err)}")
        value = None
    series["Fibre Network Degree"] = value

    try:
        value = np.real(nx.adjacency_spectrum(red_graph).max())
    except Exception as err:
        logger.debug(f"Network Eigenvalue calculation failed: {str(err)}")
        value = None
    series["Fibre Network Eigenvalue"] = value

    try:
        value = nx.algebraic_connectivity(red_graph, weight="r")
    except Exception as err:
        logger.debug(f"Network Connectivity calculation failed: {str(err)}")
        value = None
    series["Fibre Network Connectivity"] = value

    return series


def fibre_network_metrics(fibre_networks: list[FibreNetwork]) -> pd.DataFrame:
    """Analysis of list of `FibreNetwork` objects

    Parameters
    ----------
    fibre_networks : list of `<class: FibreNetwork>`
        List of fibre networks to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from networkx Graph and scikit-image
        regionprops objects
    """

    database = pd.DataFrame()

    for i, fibre_network in enumerate(fibre_networks):
        # if segment.filled_area >= 1E-2 * image_shg.size:
        series = _network_metrics(fibre_network)
        database = pd.concat(
            [database, series.to_frame(name=str(i)).T], ignore_index=True
        )

    return database
