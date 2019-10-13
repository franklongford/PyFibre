import logging
import numpy as np

from skimage.measure import regionprops
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_closing

from pyfibre.model.tools.segmentation import (
    fibre_segmentation, cell_segmentation,
    binary_to_segments
)
from pyfibre.model.tools.segment_utilities import (
    segments_to_binary, mean_binary
)


logger = logging.getLogger(__name__)


def segment_image_labels(multi_image, networks, networks_red,
                  scale=1.0):

    fibre_net_seg = fibre_segmentation(
        multi_image.image_shg, networks, networks_red)

    if multi_image.pl_analysis:

        fibre_net_binary = segments_to_binary(
            fibre_net_seg, multi_image.shape)
        fibre_filter = np.where(fibre_net_binary, 2, 0.25)
        fibre_filter = gaussian_filter(fibre_filter, 1.0)

        cell_seg, fibre_col_seg = cell_segmentation(
            multi_image.image_shg * fibre_filter,
            multi_image.image_pl, multi_image.image_tran,
            scale=scale)

        fibre_col_binary = segments_to_binary(fibre_col_seg,
                                               multi_image.shape)
        fibre_col_binary = binary_dilation(fibre_col_binary, iterations=2)
        fibre_col_binary = binary_closing(fibre_col_binary)

        fibre_binary = mean_binary(
            multi_image.image_shg, fibre_net_binary, fibre_col_binary,
            min_size=150, min_intensity=0.13)
        fibre_seg = binary_to_segments(fibre_binary,
                                       multi_image.image_shg)

    else:
        fibre_seg = fibre_net_seg

        fibre_binary = segments_to_binary(
            fibre_seg, multi_image.shape)
        cell_binary = ~fibre_binary
        cell_seg = binary_to_segments(
            cell_binary, multi_image.image_shg)

    fibre_binary = segments_to_binary(fibre_seg, multi_image.shape)
    global_binary = np.where(fibre_binary, 0, 1)
    global_seg = regionprops(global_binary, coordinates='xy')

    cell_binary = segments_to_binary(cell_seg, multi_image.shape)
    global_binary = np.where(cell_binary, 0, 1)
    global_seg += regionprops(global_binary, coordinates='xy')

    return global_seg, fibre_seg, cell_seg


def segment_image(multi_image, networks, networks_red,
                  scale=1.0):

    fibre_net_seg = fibre_segmentation(
        multi_image.image_shg, networks, networks_red)

    if multi_image.pl_analysis:

        fibre_net_binary = segments_to_binary(
            fibre_net_seg, multi_image.shape)
        fibre_filter = np.where(fibre_net_binary, 2, 0.25)
        fibre_filter = gaussian_filter(fibre_filter, 1.0)

        cell_seg, fibre_col_seg = cell_segmentation(
            multi_image.image_shg * fibre_filter,
            multi_image.image_pl, multi_image.image_tran,
            scale=scale)

        fibre_col_binary = segments_to_binary(fibre_col_seg,
                                               multi_image.shape)
        fibre_col_binary = binary_dilation(fibre_col_binary, iterations=2)
        fibre_col_binary = binary_closing(fibre_col_binary)

        fibre_binary = mean_binary(
            multi_image.image_shg, fibre_net_binary, fibre_col_binary,
            min_size=150, min_intensity=0.13)

        fibre_seg = binary_to_segments(
            fibre_binary, multi_image.image_shg, 150, 0.05)
        cell_seg = binary_to_segments(
            ~fibre_binary, multi_image.image_pl, 250, 0.01)

    else:
        fibre_seg = fibre_net_seg
        fibre_binary = segments_to_binary(
            fibre_seg, multi_image.shape)
        cell_seg = binary_to_segments(
            ~fibre_binary, multi_image.image_pl, 250, 0.01
        )

    fibre_binary = segments_to_binary(fibre_seg, multi_image.shape)
    global_binary = np.where(fibre_binary, 0, 1)
    global_seg = regionprops(global_binary, coordinates='xy')

    cell_binary = segments_to_binary(cell_seg, multi_image.shape)
    global_binary = np.where(cell_binary, 0, 1)
    global_seg += regionprops(global_binary, coordinates='xy')

    return global_seg, fibre_seg, cell_seg
