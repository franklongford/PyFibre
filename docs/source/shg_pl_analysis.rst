Automatic SHG and PL Image Detection
------------------------------------

If you have performed multiple imaging techniques on the same region, then PyFibre is able to take advantage of this to
provide extra analysis. Currently both Second Harmonic Generation (SHG) and Photoluminescence (PL) imaging techniques
are supported.

Loading images containing the keyword ``SHG`` or ``PL`` (non case sensitive) in the file path (see below) will allow
PyFibre automatically match these up based on the ``{prefix}``::

    {directory}/{prefix}-{keyword}{suffix}.tif

The files will then appear as::

    {directory}/{prefix}


Composite RGB Image
-------------------
Both SHG and PL imaging techniques produce greyscale images, so a way to combine them into a composite RGB data format
must be developed first.

We create a composite 3 channel image from the SHG, PL and PL transmission data, corresponding to red, blue and green
channels respectively, taken at a fixed biopsy region. Each RGB pixel vector is then normalised into a unit vector,
in order to reduce the dependence on lighting intensity of each image type. Therefore we expect fiborous features
to show up as unit vectors with large red components, whereas cellular features should result in unit vectors containing
prominently large green and blue components.

Colour Clustering
~~~~~~~~~~~~~~~~~

After identification of the main 8 colour clusters present in our RGB composite image, we then proceed to assign whether these
clusters should be considered either cellular or fibrous, based on their centroid unit vector :math:`\mathbf{c}` and
average non-zero intensity :math:`\bar{I}` values. We consider the cluster to contain cellular features if it
contains a non-zero value of :math:`\phi(\mathbf{c}, \bar{I})`. The terms in :math:`\phi(\mathbf{c}, \bar{I})` consist
of 3 angles on the RGB colour sphere corresponding to the unit vector :math:`\mathbf{c}`  as well as :math:`\bar{I}`.
We define a simple vector :math:`v(r, g, b, i)` (equation :eq:`kmeans_cluster`) that indicates the boundary between
cellular and fibrous domains; a cluster is only considered part of the cellular region if each component has a value
lower than this boundary.

.. math::
    :label: kmeans_cluster
    :nowrap:

    \begin{equation}
        \phi(\mathbf{c}, \bar{I}) =  \prod \left\{\begin{array}{lclcl}
        1 & if & \arcsin(\mathbf{c}_R) < v_r & else & 0\\
        1 & if & \arcsin(\mathbf{c}_G) < v_g  & else & 0\\
        1 & if & \arccos(\mathbf{c}_B) < v_b & else & 0\\
        1 & if & \bar{I} < v_i & else & 0
        \end{array}\right.
    \end{equation}

The pixels present in these accepted clusters are then combined to form our binary ""cell filter"", which we use as the
basis for further image segmentation. It should be noted that the default values of :math:`v(r, g, b, i)` used
in our software have determined heuristically, based on available data and are therefore relative, rather than absolute,
units. There are not expected to produce consistent behaviour for images from different sources.

In order to deal with the stochastic nature of the KMeans algorithm
we rank each cell cluster proportional the L1 distance of all centroids from :math:`v(r, g, b, i)`, resulting in the
cost function :math:`\Psi` (equation :eq:`kmeans_cost`). The KMeans run that creates the lowest average :math:`\Psi`
is then chosen as the optimal solution for our cell filter.

.. math::
    :label: kmeans_cost


    \begin{equation}
    \Psi = \sum\limits_i^{N} |\arcsin(\mathbf{c}_{R_i}) - v_r| +  |\arcsin(\mathbf{c}_{G_i}) - v_g| + |\arccos(\mathbf{c}_{B_i}) - v_b| + |\bar{I}_i - v_i|
    \end{equation}

Metrics
~~~~~~~

PyFibre calculates properties for the global images and each segmented region. The resultant databases for
each section are then labelled::

    {directory}/data/{prefix}_global.h5 = global image output (also in .xls format)
    {directory}/data/{prefix}_fibre.h5 = fibre segmented image output (also in .xls format)
    {directory}/data/{prefix}_network.h5 = fibre networks output (also in .xls format)
    {directory}/data/{prefix}_cell.h5 = cell segmented image output (also in .xls format)

The database metrics have the following columns:

.. csv-table:: SHG-PL-Trans Image Metrics
    :header: "Metric", "Description", "Category"
    :widths: 20, 30, 10

    "No. Fibres", "Number of extracted fibres", "Content"
    "Fibre Waviness", "Average fibre waviness (length / displacement)", "Content"
    "Fibre Lengths", "Average fibre pixel length", "Content"
    "Fibre Angles", "Average fibre angle", "Content"
    "Fibre Segment Coverage", "Ratio of image containing fibres", "Content"
    "Fibre Segment Area", "Area of image (in pixels) containing fibres", "Shape"
    "Fibre Segment Linearity", "Average ratio of diameter to perimeter for fibre segments", "Shape"
    "Fibre Segment Eccentricity", "Average eccentricity metric of fibre segments", "Shape"
    "Fibre Segment SHG Angle SDI", "Angle spectrum SDI (mean / max) for all SHG image pixels in fibre regions", "Texture"
    "Fibre Segment SHG Anisotropy", "Anisotropy of structure tensor for all SHG image pixels in fibre regions", "Texture"
    "Fibre Segment SHG Pixel Anisotropy", "Mean anisotropy of structure tensor for all SHG image pixels in fibre regions", "Texture"
    "Fibre Segment SHG Mean", "Mean pixel intensity of SHG image in fibre segment", "Texture"
    "Fibre Segment SHG STD", "Standard deviation of pixel intensity of SHG image in fibre segment", "Texture"
    "Fibre Segment SHG Entropy", "Average Shannon entropy of pixel intensities of SHG image in fibre segment", "Texture"
    "Fibre Network Degree", "Average fibre network number of edges per node", "Network"
    "Fibre Network Eigenvalue", "Max eigenvalue of network adjacency matrix", "Network"
    "Fibre Network Connectivity", "Average fibre network connectivity", "Network"
    "Fibre Network Cross-Link Density", "Average cross-links per fibre", "Content"
    "No. Cells", "Number of cell segments", Content
    "Cell Segment Coverage", "Ratio of image containing fibres/cells", "Content"
    "Cell Segment Area", "Area of image (in pixels) containing cells", "Shape"
    "Cell Segment Linearity", "Average ratio of diameter to perimeter for cell segments", "Shape"
    "Cell Segment Eccentricity", "Average eccentricity metric of cell segments", "Shape"
    "Cell Segment PL Angle SDI", "Angle spectrum SDI (mean / max) for all PL image pixels in cell regions", "Texture"
    "Cell Segment PL Anisotropy", "Anisotropy of structure tensor for all PL image pixels in cell regions", "Texture"
    "Cell Segment PL Pixel Anisotropy", "Mean anisotropy of structure tensor for all PL image pixels in cell regions", "Texture"
    "Cell Segment PL Intensity Mean", "Mean pixel intensity of PL image in cell segment", "Texture"
    "Cell Segment PL Intensity STD", "Standard deviation of pixel intensity of PL image in cell segment", Texture
    "Cell Segment PL Intensity Entropy", "Average Shannon entropy of pixel intensities of PL image in cell segment", "Texture"
