"""
PyFibre
BD RGB clustering routine

Created by: Frank Longford
Created on: 20/10/2019

Last Modified: 20/10/2019
"""

import logging
import numpy as np

from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from skimage.util import pad
from skimage.morphology import remove_small_objects
from skimage.color import rgb2grey
from skimage.exposure import equalize_hist

from sklearn.cluster import MiniBatchKMeans

from .preprocessing import clip_intensities

logger = logging.getLogger(__name__)


def create_scaled_image(image, p_intensity=(2, 98), sm_size=7):
    """Create a scaled image with enhanced and smoothed RGB
    balance"""

    image_channels = image.shape[-1]
    image_scaled = np.zeros(image.shape, dtype=int)
    pad_size = 10 * sm_size

    # Mimic contrast stretching decorrstrech routine in MatLab
    for i in range(image_channels):
        image_scaled[:, :, i] = 255 * clip_intensities(
            image[:, :, i], p_intensity=p_intensity)

    # Pad each channel, equalise and smooth to remove salt and pepper noise
    for i in range(image_channels):
        padded = pad(image_scaled[:, :, i], [pad_size, pad_size], 'symmetric')
        equalised = 255 * equalize_hist(padded)

        # Double median filter
        smoothed = median_filter(equalised, size=(sm_size, sm_size))
        smoothed = median_filter(smoothed, size=(sm_size, sm_size))

        # Transfer original image from padded back
        image_scaled[:, :, i] = smoothed[pad_size: pad_size + image.shape[0],
                                         pad_size: pad_size + image.shape[1]]

    return image_scaled


def cluster_colours(image, n_clusters=8, n_init=10):

    image_size = image.shape[0] * image.shape[1]
    image_shape = (image.shape[0], image.shape[1])
    image_channels = image.shape[-1]

    # Perform k-means clustering on PL image
    X = np.array(image.reshape((image_size, image_channels)), dtype=float)
    clustering = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init,
                                 reassignment_ratio=0.99, init_size=n_init*100,
                                 max_no_improvement=15)
    clustering.fit(X)

    labels = clustering.labels_.reshape(image_shape)
    centres = clustering.cluster_centers_

    return labels, centres


def BD_filter(image, n_runs=2, n_clusters=10, p_intensity=(2, 98),
              sm_size=5, param=(0.65, 1.1, 1.40, 0.92)):
    """Segmentation filtering using k-means clustering on RGB colour channels.
    Adapted from CurveAlign BDcreationHE routine"""

    assert image.ndim == 3

    image_size = image.shape[0] * image.shape[1]
    image_shape = (image.shape[0], image.shape[1])
    image_channels = image.shape[-1]

    image_scaled = create_scaled_image(image, p_intensity, sm_size)

    # Generate greyscale image of RGB
    greyscale = rgb2grey(image_scaled.astype(np.float64))
    greyscale /= greyscale.max()

    tot_labels = []
    tot_centres = []
    tot_cell_clusters = []
    cost_func = np.zeros(n_runs)

    for run in range(n_runs):

        logger.debug(f'BD Filter: Run {run+1} of {n_runs}')

        labels, centres = cluster_colours(image_scaled, n_clusters)
        tot_labels.append(labels)

        "Reorder labels to represent average intensity"
        intensities = np.zeros(n_clusters)

        for i in range(n_clusters):
            cluster_values = greyscale[np.where(labels == i)]
            intensities[i] = cluster_values.sum() / np.count_nonzero(cluster_values)

        magnitudes = np.sqrt(np.sum(centres**2, axis=-1))
        norm_centres = centres / np.repeat(magnitudes, image_channels).reshape(centres.shape)
        tot_centres.append(norm_centres)

        # Convert RGB centroids to spherical coordinates
        X = np.arcsin(norm_centres[:, 0])
        Y = np.arcsin(norm_centres[:, 1])
        Z = np.arccos(norm_centres[:, 2])
        I = intensities

        # Define the plane of division between cellular and fibrous clusters
        #data = np.stack((X, Y, Z, I), axis=1)
        #clusterer = KMeans(n_clusters=2)
        #clusterer.fit(data)
        #cell_clusters = clusterer.labels_

        cell_clusters = (X <= param[0]) * (Y <= param[1]) * (Z <= param[2]) * (I <= param[3])
        chosen_clusters = np.argwhere(cell_clusters).flatten()
        cost_func[run] += (X[chosen_clusters].mean() +  Y[chosen_clusters].mean()
                           + Z[chosen_clusters].mean() + I[chosen_clusters].mean())
        tot_cell_clusters.append(chosen_clusters)

    labels = tot_labels[cost_func.argmin()]
    norm_centres = tot_centres[cost_func.argmin()]
    cell_clusters = tot_cell_clusters[cost_func.argmin()]

    intensities = np.zeros(n_clusters)
    segmented_image = np.zeros((n_clusters,) + image.shape, dtype=int)
    for i in range(n_clusters):
        segmented_image[i][np.where(labels == i)] += image_scaled[np.where(labels == i)]
        intensities[i] = greyscale[np.where(labels == i)].sum() / np.where(labels == i, 1, 0).sum()

    "Select blue regions to extract epithelial cells"
    epith_cell = np.zeros(image.shape)
    for i in cell_clusters:
        epith_cell += segmented_image[i]
    epith_grey = rgb2grey(epith_cell)

    """
    "Convert RGB centroids to spherical coordinates"
    X = np.arcsin(norm_centres[:, 0])
    Y = np.arcsin(norm_centres[:, 1])
    Z = np.arccos(norm_centres[:, 2])
    I = intensities

    print(X, Y, Z, I)
    print((X <= param[0]) * (Y <= param[1]) * (Z <= param[2]) * (I <= param[3]))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.figure(100, figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    plt.figure(1000, figsize=(10, 10))
    plt.imshow(image_scaled)
    plt.axis('off')

    for i in range(n_clusters):
        plt.figure(i)
        plt.imshow(segmented_image[i])

    not_clusters = [i for i in range(n_clusters) if i not in cell_clusters]

    plt.figure(1001)
    plt.scatter(X[cell_clusters], Y[cell_clusters])
    plt.scatter(X[not_clusters], Y[not_clusters])
    for i in range(n_clusters): plt.annotate(i, (X[i], Y[i]))

    plt.figure(1002)
    plt.scatter(X[cell_clusters], Z[cell_clusters])
    plt.scatter(X[not_clusters], Z[not_clusters])
    for i in range(n_clusters): plt.annotate(i, (X[i], Z[i]))

    plt.figure(1003)
    plt.scatter(X[cell_clusters], I[cell_clusters])
    plt.scatter(X[not_clusters], I[not_clusters])
    for i in range(n_clusters): plt.annotate(i, (X[i], I[i]))

    plt.show()	
    #"""

    # Dilate binary image to smooth regions and remove small holes / objects
    epith_cell_BW = np.where(epith_grey, True, False)
    epith_cell_BW_open = binary_opening(epith_cell_BW, iterations=2)

    BWx = binary_fill_holes(epith_cell_BW_open)
    BWy = remove_small_objects(~BWx, min_size=20)

    # Return binary mask for cell identification
    mask_image = remove_small_objects(~BWy, min_size=20)

    return mask_image
