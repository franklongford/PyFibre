from pyfibre.segments.segment import Segment
from pyfibre.tools.convertors import binary_to_regions, stack_to_binary

from numpy import typing as npt


def binary_to_segments(
    binary: npt.NDArray, intensity_image=None, min_size=100, min_frac=0.1
):
    """Transform binary array into a BaseSegment instance"""

    # Create a new set of segments for each region in binary
    regions = binary_to_regions(
        binary, intensity_image=intensity_image, min_size=min_size, min_frac=min_frac
    )
    segments = [Segment(region=region) for region in regions]

    return segments


def segments_to_binary(segments: list[Segment], shape: tuple[int, ...] | None = None):
    """Transform list of BaseSegment instances into a binary array"""

    stack = [segment.to_array(shape=shape) for segment in segments]
    binary = stack_to_binary(stack)

    return binary
