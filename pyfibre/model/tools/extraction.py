import logging

import networkx as nx
import numpy as np

from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

from skimage.morphology import remove_small_objects
from skimage.transform import rescale

from pyfibre.model.tools.filters import tubeness, hysteresis
from pyfibre.utilities import clear_border

from .fibre_assignment import FibreAssignment
from .fibre_extraction import (
    FibreExtraction
)
from .fibre_utilities import (
    distance_matrix, get_edge_list, remove_redundant_nodes
)

logger = logging.getLogger(__name__)


def build_network(image, scale=1, alpha=0.5, sigma=0.5, nuc_thresh=2,
                  nuc_radius=11, lmp_thresh=0.15, angle_thresh=70, r_thresh=8,
                  ):
    """
    FIRE algorithm to extract fibre network

    Parameters
    ----------
    image:  array_like, (float); shape=(nx, ny)
        Image to perform FIRE upon
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

    "Prepare input image to gain distance matrix of foreground from background"

    image_scale = rescale(image, scale, multichannel=False,
                          mode='constant', anti_aliasing=None)
    sigma *= scale

    "Apply tubeness transform to enhance image fibres"
    image_TB = tubeness(image_scale)
    threshold = hysteresis(image_TB, alpha=alpha)
    cleaned = remove_small_objects(threshold, min_size=int(64*scale**2))
    distance = distance_transform_edt(cleaned)
    smoothed = gaussian_filter(distance, sigma=sigma)
    cleared = clear_border(smoothed)

    "Set distance thresholds for fibre iterator based on scale factor"
    nuc_thresh = np.min([nuc_thresh * scale**2, 1E-1 * scale**2 * cleared.max()])
    lmp_thresh = np.min([lmp_thresh * scale**2, 1E-1 * scale**2 * cleared[np.nonzero(cleared)].mean()])
    r_thresh = int(r_thresh * scale)
    nuc_radius = int(nuc_radius * scale)

    logger.debug("Maximum distance = {}".format(cleared.max()))
    logger.debug("Mean distance = {}".format(cleared[np.nonzero(cleared)].mean()))
    logger.debug("Using thresholds:\n nuc = {} pix\n lmp = {} pix\n angle = {} deg\n edge = {} pix".format(
            nuc_thresh, lmp_thresh, angle_thresh, r_thresh))

    fibre_network = FibreExtraction(
        nuc_thresh=nuc_thresh, lmp_thresh=lmp_thresh, angle_thresh=angle_thresh,
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
        edge_count = np.array([subgraph.degree[node] for node in subgraph], dtype=int)
        graph_check = np.sum(edge_count > 1) > 1
        graph_check *= np.sum(edge_count == 1) > 1
        if not graph_check:
            node_remove_list += list(subgraph.nodes())

    network.remove_nodes_from(node_remove_list)
    network = nx.convert_node_labels_to_integers(network)

    return network


def simplify_network(network):
    """Simplify all linear sections of network by removing nodes
    containing 2 degrees"""

    new_network = network.copy()
    edge_list = get_edge_list(new_network, max_degree=2)

    while edge_list.size > 0:
        for edge in edge_list:
            try:
                new_network = nx.contracted_edge(
                    new_network, edge, self_loops=False)
            except (ValueError, KeyError):
                pass

        edge_list = get_edge_list(new_network, max_degree=2)

    new_network = nx.convert_node_labels_to_integers(new_network)

    node_coord = [new_network.nodes[i]['xy'] for i in new_network.nodes()]
    node_coord = np.stack(node_coord)
    d_coord, r2_coord = distance_matrix(node_coord)
    r_coord = np.sqrt(r2_coord)

    for edge in new_network.edges:
        new_network[edge[0]][edge[1]]['r'] = r_coord[edge[0]][edge[1]]

    return new_network


def network_extraction(network):
    """Extract sub-networks, simplified sub-networks and Fibre objects
    from a networkx Graph generated by modified FIRE algorithm"""

    logger.debug("Extracting and simplifying fibre networks from graph")
    n_nodes = []
    networks = []
    networks_red = []
    fibres = []

    fibre_assignment = FibreAssignment()

    for i, component in enumerate(nx.connected_components(network)):

        subgraph = network.subgraph(component)

        fibre = fibre_assignment.assign_fibres(subgraph)

        if len(fibre) > 0:
            n_nodes.append(subgraph.number_of_nodes())
            networks.append(subgraph)
            networks_red.append(simplify_network(subgraph))
            fibres.append(fibre)

    "Sort segments ranked by network size"
    indices = np.argsort(n_nodes)[::-1]

    sorted_networks = [networks[i] for i in indices]
    sorted_networks_red = [networks_red[i] for i in indices]
    sorted_fibres = [fibres[i] for i in indices]

    return sorted_networks, sorted_networks_red, sorted_fibres