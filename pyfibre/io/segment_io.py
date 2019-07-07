import numpy as np

from skimage.measure import regionprops


def save_segment(segments, file_name, file_type=''):
    """Saves scikit image regions as pickled file"""

    n = len(segments)
    segment_masks = np.zeros(((n,) + segments[0]._label_image.shape),
                             dtype=int)
    for i, segment in enumerate(segments):
        segment_masks[i] += segments[i]._label_image

    try:
        np.save(f"{file_name}_{file_type}.npy", segment_masks)
    except IOError as e:
        raise IOError(
            f"Cannot save to file {file_name}"
        ) from e


def load_segment(file_name, file_type='', image=None):
    """Loads pickled scikit image regions"""

    try:
        segment_masks = np.load(f"{file_name}_{file_type}.npy")
    except IOError as e:
        raise IOError(
            f"Cannot read file {file_name}_{file_type}.npy"
        ) from e

    n = segment_masks.shape[0]
    segments = []

    for i in range(n):
        if image is not None:
            segments += regionprops(segment_masks[i],
                                    intensity_image=image)
        else:
            segments += regionprops(segment_masks[i])

    return segments
