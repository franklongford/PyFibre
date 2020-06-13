FIbeR Extraction (FIRE)
-----------------------

Fibril networks may be considered graphs, with the fibres themselves as edges between $n$ nodes at the regions they interconnect.
A graph can be described by an adjoint matrix, an :math:`n` x :math:`n` matrix that records whether an edge is present between each
node in a system. The degree of each node determines the number of edges it possesses. The FIRE (FIbeR Extraction)
algorithm is designed to trace out a network on top of an image containing fibrous textures :cite:`Stein2008_463`.
We implement a modified version that is designed to extract fibrous detail at a higher resolution than previous versions.

The algorithm works by generating a network of nodes connected by edges that represent the outlines of fibres.
A primary set of nucleation points is chosen, which become parent nodes that can then each propagate subsequent
child nodes in the nearby region. The FIRE algorithm can therefore be modified by changing any rules required to generate
the parent nodes and locate and connect subsequent child nodes.

A number of nucleation points is generated at each local maximum on our smoothed distance matrix that lies above a
threshold value (typically 2 pix). A set of parent nodes are then assigned to the pixel coordinates of these nucleation
points. Propagation of subsequent child nodes is then performed by identifying further local maxima within a given
search region of each parent node, termed Local Maximum Points (LMPs) (:numref:`figure %s <fire-figure>`). If an LMP is a
successful candidate then a new child node will be created at its location, with an edge assigned to the parent node.

.. _fire-figure:

.. figure:: _images/FIRE_lmp.pdf

    Figure taken from :cite:`Stein2008_463` representing propagation of LMP nodes from a nucleation node.

After the initial assignment of all parent and primary child nodes, propagation of the network is continued iteratively
at each LMP until no further candidates of child nodes can be found. Typically a LMP will be accepted if it already has
been assigned to an existing node or if it lies in the direction of propagation and possesses a distance matrix value
above the threshold :math:`\epsilon_0`.

PyFibre Implementation
~~~~~~~~~~~~~~~~~~~~~~

Typically a filter can be applied to the input image to enhance any tubular-like regions. We use a simple Sato or
"tubeness filter" :cite:`Sato1998_143`, based on the Hessian eigenvalues of each pixel, which has been shown
:cite:`Bredfeldt2014_16007` to have similar performance to the much more complex Curvelet transform (CT) method.

The FIRE algorithm starts by dividing the image into a binary representation of the foreground and background.
We use the hysteresis threshold method, as explained in section \ref{section:hysteresis} in order to determine which
pixels should be included in the foreground. Then a distance matrix is computed representing the number of pixels
lying between the foreground and background (see :numref:`figure %s <fire-figure>`). We then apply a Gaussian filter
in order to smooth the discrete distance matrix into continuous data, so as to reduce the number of redundant data values.

The differences between our implementation and the original FIRE algorithm include the use of an edge length threshold
:math:`r_0` and the ability to create an edge between any existing node (whereas originally only child nodes with the
same parent node could be connected). Using :math:`r_0` allows us to both move the LMPs after their original assignment
and also control the resolution at which each fibre can be traced without altering the size of the local LMP search
window :math:`B`.

Some cleaning is applied after all child nodes have been propagated in order to remove any artifacts from the FIRE
algorithm. In general, these aim to remove any redundant or unconnected nodes, as well as small (~4 nodes) networks
that are likely to correspond to noise.

#. If any 2 nodes lie within :math:`r_1` of each other, transfer edges from the node with the lowest degree to the
   node with the highest degree
#. Remove any nodes without edges
#. Remove any remaining connected networks that either contain only 1 node with 1 edge or only 1 node with 2 or more edges.

We can manipulate the raw network :math:`\mathbf{R}` in two main ways: either to provide information about individual
fibres or to give us an insight into its overall structure and connectivity. In order to do either, however,
some extra processing is required.

Assigning Fibres
~~~~~~~~~~~~~~~~

In order to calculate properties such as fibre length and waviness we need to be able to identify individual fibres
from within each network. This is not trivial and depends solely on the rules assigned to define what an appropriate
fibre is. However, we apply a very similar approach to the FIRE algorithm in order to do so.

To begin with we identify all  "external" nodes that only contain 1 edge and so therefore will definitely reside at
the start or and of any fibre. These then become our parent nodes for a fibre, and their primary child node will be
the single node that they share an edge with. The algorithm then traces back along any nodes connected to the active
child node and rebuilds a fibre based on a set of rules.


.. code-block:: python

    for parent in all_nodes:
        for child in parent.children:
            angles = [
                calc_angle(child.coord, connect.coord, parent.coord)
                for connect in child.connected_node
            ]

            angle = np.min(angles)

            if angle <= theta_0:
                next_child = child.connected_nodes[np.argmin(angles)]
                parent.children.append(next_child)

.. bibliography:: _bibs/fire-refs.bib