import os
import numpy as np

directory = os.path.dirname(os.path.realpath(__file__))


test_shg_image_path = os.path.join(
    directory, 'test-pyfibre-shg-Stack.tif')
test_pl_image_path = os.path.join(
    directory, 'test-pyfibre-pl-Stack.tif')
test_shg_pl_trans_image_path = os.path.join(
    directory, 'test-pyfibre-pl-shg-Stack.tif')

test_fibre_mask = np.load(
    os.path.join(directory, 'test-pyfibre_fibre_segments.npy')
)
