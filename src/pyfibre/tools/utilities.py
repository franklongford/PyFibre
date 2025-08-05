import logging
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation

from skimage import measure
from skimage.morphology import remove_small_objects, remove_small_holes

logger = logging.getLogger(__name__)


def bbox_indices(region):
    """Return indices identifying region bounding box"""
    minr, minc, maxr, maxc = region.bbox
    indices = np.mgrid[minr:maxr, minc:maxc]
    return indices[0], indices[1]


def bbox_sample(region, metric):
    """Extract image that lies within region bounding box

    Parameters
    ----------
    region: skimage.RegionProperties
        Region defining pixels within image to analyse
    metric: array-like
        Metric for all pixels in image to be analysed
    """

    # Identify metrics for pixels within bounding box
    indices = bbox_indices(region)

    return metric[indices]


def smooth_binary(binary, sigma=None):
    """Smooths binary image based on Gaussian filter with
     sigma standard deviation"""

    if sigma is not None:
        smoothed = gaussian_filter(
            binary.astype(float), sigma=sigma
        )
        # Convert float image back to binary
        binary = np.where(smoothed, 1, 0)

    return binary


def region_check(region, min_size=0, min_frac=0,
                 edges=False, max_x=0, max_y=0):
    """Return whether input region passes minimum area and average
    intensity checks"""

    check = True

    if edges:
        minr, minc, maxr, maxc = region.bbox
        edge_check = (minr != 0) * (minc != 0)
        edge_check *= (maxr != max_x)
        edge_check *= (maxc != max_y)

        check *= edge_check

    check *= region.filled_area >= min_size

    if region._intensity_image is not None:
        region_filter = (region.image * region.intensity_image)
        region_frac = region_filter.sum() / region.filled_area
        check *= (region_frac >= min_frac)

    return check


def region_swap(masks, images, min_sizes, min_fracs):
    """Performs a region_check on each region present in masks using images as
    intensity image. If check fails, removes region from mask and performs
    another region_check using same region with other image as
    intensity image. If this check passes, assigns region onto other mask."""

    for i, j in [[0, 1], [1, 0]]:

        labels = measure.label(masks[i].astype(np.int))

        for region_1 in measure.regionprops(
                labels, intensity_image=images[i]):

            if not region_check(region_1, min_sizes[i], min_fracs[i]):
                intensity_image = bbox_sample(region_1, images[j])
                masks[i][np.where(labels == region_1.label)] = False

                region_2 = measure.regionprops(
                    np.array(region_1.image, dtype=int),
                    intensity_image=intensity_image)[0]

                if region_check(region_2, 0, min_fracs[j]):
                    masks[j][np.where(labels == region_1.label)] = True


def mean_binary(binaries, image, iterations=1, min_intensity=0,
                area_threshold=0, sigma=None):
    """Compares two binary of image and produces a
    filter based on the overlap"""

    intensity_map = image * np.mean(binaries, axis=0)
    intensity_mask = np.where(intensity_map > min_intensity, True, False)

    # Remove small holes and objects from masks
    intensity_mask = remove_small_holes(
        intensity_mask, area_threshold=area_threshold)
    intensity_mask = remove_small_objects(
        intensity_mask, min_size=area_threshold)

    # Dilate image
    binary = binary_dilation(intensity_mask, iterations=iterations)

    smooth_binary(binary, sigma)

    return binary.astype(int)
