The PyFibre package includes python implementations of some powerful analysis algorithms for fibular and cellular images.
Two main examples provided are used to map out fibrous regions as a network and identify cells from RGB colour clustering.

Network Segmentation
--------------------
Once the global fibre network has been generated using FIRE, it is possible to segment the image based on regions of
fully connected sub-networks. It its not guaranteed that the global network will be fully connected (i.e. a path can be
drawn between any 2 nodes), since each nucleation point is propagated independently. Therefore it becomes possible to
treat each connected sub-network separately.

The segmentation process begins by creating a binary matrix containing straight lines between each network node and
performing a binary dilation for a set number of iterations. A Gaussian filter is then applied in order to smooth the
fibre edges.

Colour Segmentation
-------------------

An alternative method of segmentation uses clustering of RGB values in a colour image. This is commonly used for
microscope-stained HE images in software packages such as CurveAlign :cite:`Liu2017`, since typical gram-straining
yields cellular regions of a standard pigment.

In order to enhance the clustering of colours and yield smooth segments we apply some smoothing techniques on the
raw RGB image. These techniques mimic those carried out by the CurveAlign software BDcreationHE algorithm.
To begin with we apply a contrast stretching routine to each channel in order to maximise their differences.
The channels are then padded around the outside with a region of zeroed pixels and a intensity histogram equalisation
algorithm is applied, as detailed in section \ref{section:equalisation}. Finally, a median filter is used to smooth
the image, before the extra padded pixels are removed. The effect of this procedure is demonstrated in figure
\ref{fig:composite}.

Kmeans Clustering
~~~~~~~~~~~~~~~~~

Segmentation of the image is performed by clustering of each pixel's RGB components, independent of its location.
Considering that a typical biopsy image contains $512\times512=262144$ pixels and therefore $512\times512\times3=786432$
data points, a batch implementation of any clustering algorithm needs to be used for computational efficiency.
We use the ``MiniBatchKMeans`` function as `implemented <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html>`_
in scikit-learn, which has been shown to achieve very similar performance to its full KMeans implementation :cite:`Sculley2010`.
It must be noted that KMeans is a stochastic algorithm, and therefore is not guaranteed to return the optimum solution
at every run.

Consequently, we rank each set of pixel clusters produces by the segmentation by a cost function. The KMeans
run that creates the lowest average cost is then chosen as the optimal solution for our RGB filter.

For each Kmeans cluster run:

#. Perform K-means clustering on image pixels to form :math:`n_{clusters}`
#. Classify clusters into cellular regions using :math:`cluster\_func`
#. Calculate cost function of segmentation

For all segmentations:

#. Identify segmentation with lowest cost function
#. Generate binary mask of all image pixels in cellular regions


.. bibliography:: seg-refs.bib