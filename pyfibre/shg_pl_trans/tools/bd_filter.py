import numpy as np

from pyfibre.model.tools.base_bd_filter import BaseBDFilter


def nonzero_mean(array):
    """Return mean of non-zero values"""
    return array.sum() / np.count_nonzero(array)


def spherical_coords(coords):
    """Transform cartesian coordinates to spherical

    Parameters
    ----------
    coords: array_like, shape=(..., 3)
        Values in 3D cartesian coordinates

    Returns
    -------
    x, y, z: array-like, shape=(...)
        Spherical coordinate system in 3D
    """
    x = np.arcsin(coords[..., 0])
    y = np.arcsin(coords[..., 1])
    z = np.arccos(coords[..., 2])

    return x, y, z


def distance_sum(vector, boundary):

    boundary = np.array(boundary)

    # Cost function, based on separation of all clusters
    # from boundary
    cost = np.nansum([
        np.abs(vec - boundary) for vec in vector
    ])

    return cost


def binary_classifier_spherical(centres, intensities, boundary):
    """Create new clusters from results of KMeans.
    Attempts to add regularisation parameters

    Parameters
    ----------
    centres: array-like, shape=(N, 4)
        Centres of each cluster identified by K-means filter
    intensities: array-like, shape=(N,)
        Mean image intensity values for each centre
    boundary: array-like, shape = (4,)
        Minimum values for the boundary between classifier.

    Returns
    -------
    clusters: array-like
        Indices of centres in cellular regions
    cost:
        Cost associated with segmentation
    """

    boundary = np.array(boundary)

    # Convert RGB centroids to spherical coordinates
    x, y, z = spherical_coords(centres)

    # Calculate mask and cost function associated with segmentation
    # based on distance between boundary
    vector = np.stack([x, y, z, intensities], axis=-1)

    # Identify centroids in segmentation mask
    mask = [
        all(vec <= boundary) for vec in vector
    ]

    cost = -distance_sum(vector, boundary)

    return mask, cost


class SHGPLTransBDFilter(BaseBDFilter):

    def cellular_classifier(
            self, label_image, centres,
            boundary=(0.65, 1.1, 1.40, 0.92)):
        """Classifies centroids in image into either cellular or non-cellular
        regions. This is based on boundary values"""

        n_clusters = len(centres)

        # Calculate average intensity of each labelled region
        intensities = np.zeros(n_clusters)
        for index in range(n_clusters):
            indices = np.where(label_image == index)
            cluster_values = self._greyscale[indices]
            intensities[index] += nonzero_mean(cluster_values)

        # Normalise centroids
        magnitudes = np.sqrt(np.sum(centres ** 2, axis=-1))
        magnitudes = np.repeat(magnitudes, centres.shape[-1])
        centres = centres / magnitudes.reshape(centres.shape)

        # Create cluster mask based on centre proximity to
        # boundary
        mask, cost = binary_classifier_spherical(
            centres, intensities, boundary)

        return mask, cost
