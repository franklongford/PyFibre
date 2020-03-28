import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

from skimage.morphology import local_maxima


def get_node_coord_array(graph):
    """Return a numpy array containing xy attributes of all nodes
    in graph"""

    node_coord = [graph.nodes[i]['xy'] for i in graph.nodes()]

    return np.stack(node_coord)


def check_2D_arrays(array1, array2, thresh=1):
    """Returns indices where values of array1 are within thresh distance
    of array2

    Parameters
    ----------
    array1 : array_like
        Array to be inspected
    array2 : array_like
        Array to be inspected, does not need to be same size as array1
    thresh : int, optional
        Threshold distance to return the corresponding array indices

    Returns
    --------
    array1_indices: array_like of int
        Indices of array1 that meet threshold critera
    array2_indices: array_like of int
        Indices of array2 that meet threshold critera
    """

    shape = (array1.shape[0], array2.shape[0], 2)

    array1_mat = np.tile(array1, (1, array2.shape[0]))
    array1_mat = array1_mat.reshape(shape)

    array2_mat = np.tile(array2, (array1.shape[0], 1))
    array2_mat = array2_mat.reshape(shape)

    diff = np.sum((array1_mat - array2_mat)**2, axis=2)
    array1_indices = np.argwhere(diff <= thresh**2)[:, 0]
    array2_indices = np.argwhere(diff <= thresh**2)[:, 1]

    return array1_indices, array2_indices


def cos_sin_theta_2D(vector, r_vector):
    """
    Returns cosine and sine of angles of intersecting vectors
    between even and odd indices

    Parameters
    ----------
    vector:  array_like, (float); shape=(n_vector, n_dim)
        Array of displacement vectors between connecting beads
    r_vector: array_like, (float); shape=(n_vector)
        Array of radial distances between connecting beads

    Returns
    -------
    cos_the:  array_like (float); shape=(n_vector/2)
        Cosine of the angle between each pair of displacement
        vectors
    sin_the: array_like (float); shape=(n_vector/2)
        Sine of the angle between each pair of displacement
        vectors
    r_prod: array_like (float); shape=(n_vector/2)
        Product of radial distance between each pair of
        displacement vectors
    """

    n_vector = vector.shape[0]
    n_dim = vector.shape[1]

    temp_vector = np.reshape(vector, (n_vector // 2, 2, n_dim))

    # Calculate |rij||rjk| product for each pair of vectors
    r_prod = np.prod(
        np.reshape(r_vector, (n_vector // 2, 2)), axis=1)

    # Form dot product of each vector pair rij*rjk in vector
    # array corresponding to an angle
    dot_prod = np.sum(np.prod(temp_vector, axis=1), axis=1)

    # Calculate cos(theta) for each angle
    cos_the = dot_prod / r_prod

    return cos_the


def reduce_coord(coord, values, thresh=1):
    """
    Find elements in coord that lie within thresh distance of
    each other. Remove element with lowest corresponding value.
    """

    if coord.shape[0] <= 1:
        return coord

    thresh = np.sqrt(2 * thresh**2)
    r_coord = cdist(coord, coord)

    del_coord = np.argwhere(
        (r_coord <= thresh) - np.identity(coord.shape[0]))
    del_coord = del_coord[np.arange(0, del_coord.shape[0], 2)]
    indices = np.stack((values[del_coord[:, 0]],
                        values[del_coord[:, 1]])).argmax(axis=0)
    del_coord = [a[i] for a, i in zip(del_coord, indices)]

    coord = np.delete(coord, del_coord, 0)

    return coord


def new_branches(image, coord, ring_filter, max_thresh=0.2):
    """Find local maxima in image within max_thresh of coord,
    excluding pixels in ring filter"""

    filtered = image * ring_filter
    branch_coord = np.argwhere(
        local_maxima(filtered) * image >= max_thresh)
    branch_coord = reduce_coord(
        branch_coord, image[branch_coord[:, 0], branch_coord[:, 1]])

    n_branch = branch_coord.shape[0]
    branch_vector = np.tile(coord, (n_branch, 1)) - branch_coord
    branch_r = np.sqrt(np.sum(branch_vector**2, axis=1))

    return branch_coord, branch_vector, branch_r


def branch_angles(direction, branch_vector, branch_r):

    n_branch = branch_vector.shape[0]
    dir_vector = np.tile(direction, (n_branch, 1))
    dir_r = np.ones(n_branch)

    combined_vector = np.hstack(
        (branch_vector, dir_vector)).reshape(n_branch * 2, 2)
    combined_r = np.column_stack((branch_r, dir_r)).flatten()
    cos_the = cos_sin_theta_2D(combined_vector, combined_r)

    return cos_the


def transfer_edges(network, source, target):
    """Transfer edges from source node to target"""

    for node in list(network.adj[source]):
        network.remove_edge(node, source)

        if node != target:
            network.add_edge(node, target)
            diff = network.nodes[target]['xy'] - network.nodes[node]['xy']
            network[node][target]['r'] = np.sqrt((diff ** 2).sum())


def distance_matrix(node_coord):
    """Calculate distances between each index value of
    node_coord"""

    node_coord_matrix = np.tile(node_coord, (node_coord.shape[0], 1))
    node_coord_matrix = node_coord_matrix.reshape(
        node_coord.shape[0], node_coord.shape[0], node_coord.shape[1])
    d_node = node_coord_matrix - np.transpose(node_coord_matrix, (1, 0, 2))
    r2_node = np.sum(d_node**2, axis=2)

    return d_node, r2_node.astype(float)


def get_edge_list(graph, max_degree=2):
    """Get a list of edges between nodes that contain less that
    max_degree edges"""

    edge_list = np.empty((0, 2), dtype=int)

    for edge in graph.edges:

        edge = np.array(edge)
        degrees = np.array((graph.degree[edge[0]], graph.degree[edge[1]]))

        degree_check = np.any(degrees != 1)
        degree_check *= np.all(degrees <= max_degree)

        if degree_check:
            order = np.argsort(degrees)
            edge_list = np.concatenate(
                (edge_list,
                 np.expand_dims(edge[order], axis=0))
            )

    return edge_list


def remove_redundant_nodes(network, r_thresh=2):
    """Reduces any two nodes that are within r_thresh distance of each
    other down to one, by transferring any edges on the least connected node
    to the most connected node before removing the least connected node"""

    network = nx.convert_node_labels_to_integers(network)
    r2_thresh = r_thresh**2
    checking = True

    while checking:
        # Calculate distances between each node
        node_coord = [network.nodes[i]['xy'] for i in network.nodes()]
        node_coord = np.stack(node_coord)
        d_coord, r2_coord = distance_matrix(node_coord)

        # Deal with one set of coordinates
        upper_diag = np.triu_indices(r2_coord.shape[0])
        r2_coord[upper_diag] = r2_thresh

        # Find nodes in a similar location
        duplicate_nodes = np.where(r2_coord < r2_thresh)
        checking = (duplicate_nodes[0].size > 0)

        # Iterate through each duplicate and transfer edges on to
        # most connected
        for node1, node2 in zip(*duplicate_nodes):
            if network.degree[node1] > network.degree[node2]:
                transfer_edges(network, node1, node2)
            elif network.degree[node1] + network.degree[node2] != 0:
                transfer_edges(network, node2, node1)

        if len(network.edges) == 0:
            # If no edges are left, reduce the network down to a
            # single node
            checking = False
            remove_list = list(network.nodes)[1:]
        else:
            # Otherwise, remove all nodes with no edges
            remove_list = list(nx.isolates(network))

        network.remove_nodes_from(remove_list)
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
    node_coord = get_node_coord_array(new_network)

    d_coord, r2_coord = distance_matrix(node_coord)
    r_coord = np.sqrt(r2_coord)

    for edge in new_network.edges:
        new_network[edge[0]][edge[1]]['r'] = r_coord[edge[0]][edge[1]]

    return new_network
