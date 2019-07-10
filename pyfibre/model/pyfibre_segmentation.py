import logging
import numpy as np

from skimage.measure import regionprops
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_closing

from pyfibre.model.tools.segmentation import (
    fibre_segmentation, cell_segmentation, create_binary_image,
    mean_binary, get_segments
)

logger = logging.getLogger(__name__)


def segment_image(multi_image, networks, networks_red,
                  scale=1.0):

    fibre_net_seg = fibre_segmentation(
        multi_image.image_shg, networks, networks_red)

    if multi_image.pl_analysis:

        fibre_net_binary = create_binary_image(
            fibre_net_seg, multi_image.shape)
        fibre_filter = np.where(fibre_net_binary, 2, 0.25)
        fibre_filter = gaussian_filter(fibre_filter, 1.0)

        cell_seg, fibre_col_seg = cell_segmentation(
            multi_image.image_shg * fibre_filter,
            multi_image.image_pl, multi_image.image_tran, scale=scale)

        fibre_col_binary = create_binary_image(fibre_col_seg, multi_image.shape)
        fibre_col_binary = binary_dilation(fibre_col_binary, iterations=2)
        fibre_col_binary = binary_closing(fibre_col_binary)

        fibre_binary = mean_binary(
            multi_image.image_shg, fibre_net_binary, fibre_col_binary,
            min_size=150, min_intensity=0.13)

        fibre_seg = get_segments(
            multi_image.image_shg, fibre_binary, 150, 0.05)
        cell_seg = get_segments(
            multi_image.image_pl, ~fibre_binary, 250, 0.01)

    else:
        fibre_seg = fibre_net_seg
        cell_seg = fibre_net_seg

    fibre_binary = create_binary_image(fibre_seg, multi_image.shape)
    global_binary = np.where(fibre_binary, 0, 1)
    global_seg = regionprops(global_binary, coordinates='xy')

    cell_binary = create_binary_image(cell_seg, multi_image.shape)
    global_binary = np.where(cell_binary, 0, 1)
    global_seg += regionprops(global_binary, coordinates='xy')

    return global_seg, fibre_seg, cell_seg
