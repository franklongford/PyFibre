from pyfibre.model.tools.figures import (
    create_figure,
    create_tensor_image,
    create_network_image,
    create_region_image)


def create_shg_figures(multi_image, figname, network_graphs=None,
                       fibre_graphs=None, fibre_regions=None):
    """Creates and saves figures associated with SHG images"""

    image = multi_image.shg_image

    create_figure(image, figname + '_SHG', cmap='binary_r')
    tensor_image = create_tensor_image(image)
    create_figure(tensor_image, figname + '_tensor')

    if network_graphs is not None:
        network_image = create_network_image(image, network_graphs)
        create_figure(network_image, figname + '_network')

    if fibre_graphs is not None:
        fibre_image = create_network_image(image, fibre_graphs, 1)
        create_figure(fibre_image, figname + '_fibre')

    if fibre_regions is not None:
        fibre_region_image = create_region_image(image, fibre_regions)
        create_figure(fibre_region_image, figname + '_fibre_seg')


def create_shg_pl_trans_figures(multi_image, figname, network_graphs=None,
                                fibre_graphs=None, fibre_regions=None,
                                cell_regions=None):
    """Creates and saves figures associated with SHG-PL-Trans images"""
    create_shg_figures(multi_image, figname,
                       network_graphs, fibre_graphs, fibre_regions)

    create_figure(multi_image.pl_image, figname + '_PL', cmap='binary_r')
    create_figure(multi_image.trans_image, figname + '_trans', cmap='binary_r')

    if cell_regions is not None:

        cell_region_image = create_region_image(
            multi_image.pl_image, cell_regions)
        create_figure(cell_region_image, figname + '_cell_seg')
