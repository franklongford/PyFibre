import logging
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_closing

from pyfibre.model.objects.cell import Cell
from pyfibre.model.tools.segmentation import (
    rgb_segmentation
)
from pyfibre.model.tools.segment_utilities import (
    mean_binary)
from pyfibre.model.tools.convertors import segments_to_binary, binary_to_segments, networks_to_segments

logger = logging.getLogger(__name__)


def cell_segmentation(multi_image, fibre_networks, scale=1.0, pl_analysis=False):

    fibre_segments = [fibre_network.segment
                      for fibre_network in fibre_networks]

    if pl_analysis:
        # Create a filter for the SHG image that enhances the segments
        # identified by the FIRE algorithm
        fibre_net_binary = segments_to_binary(
            fibre_segments, multi_image.shape
        )
        fibre_filter = np.where(fibre_net_binary, 2, 0.25)
        fibre_filter = gaussian_filter(fibre_filter, 0.5)

        # Segment the PL image using k-means clustering
        cell_segments, fibre_col_seg = rgb_segmentation(
            multi_image.shg_image * fibre_filter,
            multi_image.pl_image,
            multi_image.trans_image,
            scale=scale)

        # Create a filter for the SHG image that enhances the segments
        # identified by the k-means clustering algorithm
        fibre_col_binary = segments_to_binary(
            fibre_col_seg,  multi_image.shape
        )
        fibre_col_binary = binary_dilation(fibre_col_binary,
                                           iterations=2)
        fibre_col_binary = binary_closing(fibre_col_binary)

        # Generate a filter for the SHG image that combines information
        # from both FIRE and k-means algorithms
        fibre_binary = mean_binary(
            np.array([fibre_net_binary, fibre_col_binary]),
            multi_image.shg_image,
            min_intensity=0.13)

        # Create a new set of segments for each cell region
        image = multi_image.pl_image
        cell_binary = ~fibre_binary

    else:
        # Create a filter for the PL image that corresponds
        # to the regions that have not been identified as fibrous
        # segments
        fibre_binary = segments_to_binary(
            fibre_segments, multi_image.shape
        )
        image = multi_image.shg_image
        cell_binary = ~fibre_binary

    # Segment the image using this filter
    cell_segments = binary_to_segments(
        cell_binary, image, 250, 0.01
    )

    cells = [Cell(segment=cell_segment,
                  image=image)
             for cell_segment in cell_segments]

    return cells
