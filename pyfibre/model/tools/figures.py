"""
ColECM: Collagen ExtraCellular Matrix Simulation
ANALYSIS ROUTINE

Created by: Frank Longford
Created on: 09/03/2018

Last Modified: 19/04/2018
"""

import numpy as np

from skimage import draw
from skimage.transform import rotate
from skimage.color import label2rgb, grey2rgb, rgb2hsv, hsv2rgb

from pyfibre.model.tools.filters import form_structure_tensor
from pyfibre.model.tools.analysis import tensor_analysis

BASE_COLOURS = {
    'b': (0, 0, 1),
    'g': (0, 0.5, 0),
    'r': (1, 0, 0),
    'c': (0, 0.75, 0.75),
    'm': (0.75, 0, 0.75),
    'y': (0.75, 0.75, 0),
    'k': (0, 0, 0)}


def create_figure(image, filename, figsize=(10, 10),
                  ext='png', cmap='viridis'):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)

    if image.ndim == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename + '.' + ext)
    plt.close()


def create_hsb_image(image, hue, saturation=1, brightness=1):
    """ Add color of the given hue to an greyscale image.

    By default, set the saturation to 1 so that the colors pop!
    """
    rgb = grey2rgb(image)
    hsv = rgb2hsv(rgb)

    hsv[..., 0] = hue
    hsv[..., 1] = saturation
    hsv[..., 2] = brightness

    return hsv2rgb(hsv)


def create_angle_reference(size, min_n=50, max_n=120):

    # Make circular test image
    size = np.max(np.asarray([min_n, size], dtype=int))
    size = np.min(np.asarray([size, max_n], dtype=int))

    if np.mod(size, 2) != 0:
        size += 1

    image_grid = np.mgrid[: size, : size]

    for i in range(2):
        image_grid[i] -= size * np.array(
            2 * image_grid[i] / size, dtype=int)
        image_grid[i] = np.fft.fftshift(image_grid[i])

    image_radius = np.sqrt(np.sum(image_grid ** 2, axis=0))
    image_sine = np.sin(4 * np.pi * image_radius / size)
    image_cos = np.cos(4 * np.pi * image_radius / size)
    image_rings = image_sine * image_cos

    j_tensor = form_structure_tensor(image_rings, sigma=1.0)
    pix_j_anis, pix_j_angle, pix_j_energy = tensor_analysis(j_tensor)

    pix_j_angle = rotate(pix_j_angle, 90)[: size // 2]
    pix_j_anis = pix_j_anis[: size // 2]
    pix_j_energy = np.where(
        image_radius < (size / 2), pix_j_energy, 0)[: size // 2]

    return pix_j_angle, pix_j_anis, pix_j_energy


def create_tensor_image(image, min_N=50):

    # Form nematic and structure tensors for each pixel
    j_tensor = form_structure_tensor(image, sigma=1.0)

    # Perform anisotropy analysis on each pixel
    pix_j_anis, pix_j_angle, pix_j_energy = tensor_analysis(j_tensor)

    hue = (pix_j_angle + 90) / 180
    saturation = pix_j_anis / pix_j_anis.max()
    brightness = image / image.max()

    size = 0.2 * np.sqrt(image.size)

    if size >= min_N:
        ref_angle, ref_anis, ref_energy = create_angle_reference(size)
        size = ref_angle.shape[1]
        start = - size // 2
        end = image.shape[0]

        hue[start: end, : size] = (ref_angle + 90) / 180
        saturation[start: end, : size] = ref_anis / ref_anis.max()
        brightness[start: end, : size] = ref_energy / ref_energy.max()

    # Form structure tensor image
    rgb_image = create_hsb_image(
        image, hue, saturation, brightness)

    return rgb_image


def create_region_image(image, regions):
    """Plots a figure showing identified regions

    Parameters
    ----------
    image:  array_like (float); shape=(n_x, n_y)
        Image under analysis of pos_x and pos_y
    regions:  list (skimage.region)
        List of segmented regions
    """

    image /= image.max()
    label_image = np.zeros(image.shape, dtype=int)
    label = 1

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        indices = np.mgrid[minr:maxr, minc:maxc]

        label_image[(indices[0], indices[1])] += region.image * label
        label += 1

    image_label_overlay = label2rgb(
        label_image, image=image, bg_label=0,
        image_alpha=0.99, alpha=0.25, bg_color=(0, 0, 0))

    return image_label_overlay


def create_network_image(image, networks, c_mode=0):
    """Create image with overlayed fibre networks"""

    colours = list(BASE_COLOURS.keys())

    rgb_image = grey2rgb(image)

    for j, network in enumerate(networks):

        if c_mode == 0:
            colour = BASE_COLOURS['r']
        else:
            colour = BASE_COLOURS[colours[j % len(colours)]]

        node_coord = [network.nodes[i]['xy'] for i in network]
        node_coord = np.stack(node_coord)

        mapping = zip(network.nodes,
                      np.arange(network.number_of_nodes()))
        mapping_dict = dict(mapping)

        for n, node1 in enumerate(network):
            for node2 in list(network.adj[node1]):
                m = mapping_dict[node2]
                rr, cc, val = draw.line_aa(
                    node_coord[m][0], node_coord[m][1],
                    node_coord[n][0], node_coord[n][1])

                for i, c in enumerate(colour):
                    rgb_image[rr, cc, i] = c * val * 255.9999

    return rgb_image
