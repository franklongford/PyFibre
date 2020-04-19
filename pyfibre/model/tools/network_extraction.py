import logging

import networkx as nx
import numpy as np

from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

from skimage.morphology import remove_small_objects
from skimage.transform import rescale

from pyfibre.model.objects.fibre_network import FibreNetwork
from pyfibre.model.tools.filters import tubeness, hysteresis
from pyfibre.utilities import clear_border

from .fire_algorithm import FIREAlgorithm
from .fibre_utilities import remove_redundant_nodes

logger = logging.getLogger(__name__)


def build_network(image, scale=1, alpha=0.5, sigma=0.5, nuc_thresh=2,
                  nuc_radius=11, lmp_thresh=0.15, angle_thresh=70,
                  r_thresh=7):
    """
    Uses the FibeR Extraction algorithm to extract a fibre network from
    provided image

    Parameters
    ----------
    image:  array_like, (float); shape=(nx, ny)
        Image to perform FIRE upon
    scale: float
        Scaling factor to apply to image before performing algorithm
    alpha: float
        Alpha metric to use in hysteresis threshold algorithm
    sigma: float
        Gaussian standard deviation to filter distance image
    nuc_thresh: float
        Minimum distance pixel threshold to be classed as nucleation point
    nuc_radius: float
        Minimum pixel radii between nucleation points
    lmp_thresh: float
        Minimum distance pixel threshold to be classed as lmp point
    angle_thresh: float
        Maximum angular deviation of new lmp from fibre trajectory
    r_thresh: float
        Maximum length of edges between nodes

    Returns
    -------

    network: nx.Graph
        Networkx graph object representing fibre network

    """

    # Prepare input image to gain distance matrix of foreground
    # from background"

    image_scale = rescale(image, scale, multichannel=False,
                          mode='constant', anti_aliasing=None)
    sigma *= scale

    # Apply tubeness transform to enhance image fibres"
    image_TB = tubeness(image_scale)
    threshold = hysteresis(image_TB, alpha=alpha)
    cleaned = remove_small_objects(threshold, min_size=int(64*scale**2))
    distance = distance_transform_edt(cleaned)
    smoothed = gaussian_filter(distance, sigma=sigma)
    cleared = clear_border(smoothed)

    # Set distance thresholds for fibre iterator based on scale factor"
    nuc_thresh = np.min(
        [nuc_thresh * scale**2,
         1E-1 * scale**2 * cleared.max()])
    lmp_thresh = np.min(
        [lmp_thresh * scale**2,
         1E-1 * scale**2 * cleared[np.nonzero(cleared)].mean()])
    r_thresh = int(r_thresh * scale)
    nuc_radius = int(nuc_radius * scale)

    logger.debug(
        "Maximum distance = {}".format(cleared.max()))
    logger.debug(
        f"Mean distance = {cleared[np.nonzero(cleared)].mean()}")
    logger.debug(
        "Using thresholds:\n"
        " nuc = {} pix\n "
        "lmp = {} pix\n "
        "angle = {} deg\n "
        "edge = {} pix".format(
            nuc_thresh, lmp_thresh, angle_thresh, r_thresh))

    fibre_network = FIREAlgorithm(
        nuc_thresh=nuc_thresh, lmp_thresh=lmp_thresh,
        angle_thresh=angle_thresh,
        r_thresh=r_thresh, nuc_radius=nuc_radius)
    network = fibre_network.create_network(cleared)

    # Rescale all node coordinates and edge radii
    for node in network.nodes():
        network.nodes[node]['xy'] = np.array(
            network.nodes[node]['xy'] // scale, dtype=int)

    for edge in network.edges():
        network.edges[edge]['r'] *= 1. / scale

    network = clean_network(network)

    return network


def clean_network(network, r_thresh=2):
    """Cleans network by removing isolated nodes, combining any
    two nodes that are located too close together into one, and removing
    any components that are too small to be considered fibres"""

    logger.debug("Remove all isolated nodes")
    network.remove_nodes_from(list(nx.isolates(network)))
    network = nx.convert_node_labels_to_integers(network)

    logger.debug("Checking for redundant nodes")
    network = remove_redundant_nodes(network, r_thresh)

    # Remove graph components containing either only 1 node with 1 edge or
    # 1 node with more than 1 edge"
    node_remove_list = []
    for i, component in enumerate(nx.connected_components(network)):
        subgraph = network.subgraph(component)
        edge_count = np.array(
            [subgraph.degree[node] for node in subgraph],
            dtype=int)
        graph_check = np.sum(edge_count > 1) > 1
        graph_check *= np.sum(edge_count == 1) > 1
        if not graph_check:
            node_remove_list += list(subgraph.nodes())

    network.remove_nodes_from(node_remove_list)
    network = nx.convert_node_labels_to_integers(network)

    return network


def fibre_network_assignment(network):
    """Extract sub-networks, simplified sub-networks and Fibre objects
    from a networkx Graph generated by modified FIRE algorithm"""

    logger.debug("Extracting and simplifying fibre networks from graph")

    fibre_networks = []

    for i, component in enumerate(nx.connected_components(network)):

        subgraph = network.subgraph(component)

        fibre_network = FibreNetwork(graph=subgraph)

        fibre_network.fibres = fibre_network.generate_fibres()

        if len(fibre_network.fibres) > 0:
            fibre_network.red_graph = fibre_network.generate_red_graph()
            fibre_networks.append(fibre_network)

    # Sort segments ranked by graph size
    fibre_networks = sorted(
        fibre_networks, key=lambda network: network.number_of_nodes)

    return fibre_networks
