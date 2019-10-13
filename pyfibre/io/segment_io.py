import numpy as np

from pyfibre.model.tools.segment_utilities import (
    segments_to_binary, binary_to_segments
)


def save_segment(segments, file_name, shape, file_type=None):
    """Saves scikit image regions as pickled file"""

    binary = segments_to_binary(segments, shape)

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        np.save(f"{file_name}.npy", binary)
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}"
        ) from e


def load_segment(file_name, file_type=None, image=None):
    """Loads pickled scikit image regions"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        binary = np.load(f"{file_name}.npy")
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}.npy"
        ) from e

    segments = binary_to_segments(binary, intensity_image=image)

    return segments
