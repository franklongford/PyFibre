import numpy as np

from skimage.measure import regionprops


def save_segment(segments, file_name, file_type=None):
    """Saves scikit image regions as pickled file"""

    n = len(segments)
    if n == 0:
        return

    segment_masks = np.zeros(
        ((n,) + segments[0]._label_image.shape),
        dtype=int
    )

    for i, segment in enumerate(segments):
        segment_masks[i] += np.where(
            segments[i]._label_image == segments[i].label, 1, 0
        )

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        np.save(f"{file_name}.npy", segment_masks)
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}"
        ) from e


def load_segment(file_name, file_type=None, image=None):
    """Loads pickled scikit image regions"""

    if file_type is not None:
        file_name = '_'.join([file_name, file_type])

    try:
        segment_masks = np.load(f"{file_name}.npy")
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}.npy"
        ) from e

    n = segment_masks.shape[0]
    segments = []

    for i in range(n):
        segments += regionprops(segment_masks[i],
                                intensity_image=image)

    return segments
