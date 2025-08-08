import json
import logging
import os

import numpy as np
from tifffile import TiffFile

logger = logging.getLogger(__name__)


def lookup_page(tiff_page):
    """Obtain relevant information from a TiffPage object"""

    xy_dim = (tiff_page.imagewidth, tiff_page.imagelength)
    return xy_dim, tiff_page.description


def get_fluoview_param(description, xy_dim, shape):
    desc_list = description.split("\n")
    channel_lines = [line.strip() for line in desc_list if "Gamma" in line]
    n_modes = len(channel_lines)

    # If number of modes is not in image shape (typically
    # because the image only contains one mode)
    if shape.count(n_modes) == 0:
        if n_modes == 1 and len(shape) == 2:
            minor_axis = None
        elif n_modes == 1 and len(shape) == 3:
            minor_axis = np.argmin(shape)
        else:
            raise IndexError(f"Image shape {shape} not supported")

        return minor_axis, n_modes, xy_dim

    # If there is an exact match in the image shape, identify
    # this as the axis containing each mode
    if shape.count(n_modes) == 1:
        major_axis = shape.index(n_modes)

    # If multiple image dimensions share the same number of
    # elements as number of modes, identify which corresponds
    # to each mode
    else:
        if shape[0] == n_modes:
            major_axis = 0
        else:
            raise IndexError(f"Image shape {shape} not supported")

    # Work out the minor axis (stack to average over) from the
    # remaining image dimensions
    minor_axes = [
        index
        for index, value in enumerate(shape)
        if value not in xy_dim and index != major_axis
    ]

    if len(minor_axes) == 0:
        minor_axis = None
    elif len(minor_axes) == 1:
        minor_axis = minor_axes[0]
    else:
        raise IndexError(f"Image shape {shape} not supported")

    return minor_axis, n_modes, xy_dim


def get_imagej_param(description, xy_dim, shape):
    desc_list = description.split("\n")
    slices = [line.strip() for line in desc_list if "slices" in line]

    if not slices:
        raise IndexError(f"Image shape {shape} not supported")
    else:
        n_slices = int(slices[0].split("=")[-1])
        minor_axis = shape.index(n_slices)

    # Work out the number of modes from the
    # remaining image dimensions
    n_modes = [
        index
        for index, value in enumerate(shape)
        if value not in xy_dim and index != minor_axis
    ]

    if len(n_modes) == 0:
        n_modes = 1
    elif len(n_modes) == 1:
        n_modes = shape[n_modes[0]]
    else:
        raise IndexError(f"Image shape {shape} not supported")

    return minor_axis, n_modes, xy_dim


def get_tiff_param(tiff_file: TiffFile):
    """Obtain relevant parameters of TiffFile object"""

    xy_dim, description = lookup_page(tiff_file.pages[0])
    shape = tiff_file.asarray().shape

    if tiff_file.is_fluoview:
        return get_fluoview_param(description, xy_dim, shape)

    elif tiff_file.is_imagej:
        return get_imagej_param(description, xy_dim, shape)

    else:
        # We are using test data
        desc_dict = json.loads(description)

        minor_axis = desc_dict["minor_axis"]
        n_modes = desc_dict["n_modes"]
        xy_dim = tuple(desc_dict["xy_dim"])

        return minor_axis, n_modes, xy_dim


def get_accumulation_number(file_name):
    """Extract accumulation from file name if present.
    Return default value of 1 if not present.

    Parameters
    ----------
    file_name: str
        File name of Tiff image

    Returns
    -------
    acc_number: int
        Accumulation number for image

    Notes
    -----
    Expects the following file formatting:

        <prefix>-acc<number>.ext
    """
    path, ext = os.path.splitext(file_name)
    if "acc" in path.lower():
        _, number = path.split("acc")
        return int(number)
    return 1
