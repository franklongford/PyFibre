import networkx as nx
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix
from skimage.measure import shannon_entropy

from pyfibre.model.tools.analysis import tensor_analysis, angle_analysis
from pyfibre.model.tools.feature import greycoprops_edit
from pyfibre.model.tools.filters import form_structure_tensor


def nematic_tensor_metrics(segment, nematic_tensor, tag=''):
    """Analysis of the nematic tensor"""

    database = pd.Series(dtype=object)

    minr, minc, maxr, maxc = segment.bbox
    indices = np.mgrid[minr:maxr, minc:maxc]
    segment_n_tensor = nematic_tensor[(indices[0], indices[1])]

    (segment_anis_map,
     segment_angle_map,
     segment_angle_map) = tensor_analysis(segment_n_tensor)

    segment_anis, _, _ = tensor_analysis(np.mean(segment_n_tensor, axis=(0, 1)))

    database[f"{tag} Angle SDI"], _ = angle_analysis(segment_angle_map, segment_anis_map)
    database[f"{tag} Anisotropy"] = segment_anis[0]
    database[f"{tag} Pixel Anisotropy"] = np.mean(segment_anis_map)

    return database


def region_shape_metrics(region, tag=''):
    """Analysis for a scikit-image region"""

    database = pd.Series(dtype=object)

    # Perform all non-intensity image relevant metrics
    database[f"{tag} Area"] = region.area
    database[f"{tag} Linearity"] = 1 - np.pi * region.equivalent_diameter / region.perimeter
    database[f"{tag} Eccentricity"] = region.eccentricity
    database[f"{tag} Coverage"] = region.extent

    segment_hu = region.moments_hu
    database[f"{tag} Hu Moment 1"] = segment_hu[0]
    database[f"{tag} Hu Moment 2"] = segment_hu[1]
    database[f"{tag} Hu Moment 3"] = segment_hu[2]
    database[f"{tag} Hu Moment 4"] = segment_hu[3]

    return database


def region_texture_metrics(region, image=None, tag=''):

    database = pd.Series(dtype=object)

    # Check to see whether intensity_image is present or image argument
    # has been supplied
    if image is not None:
        minr, minc, maxr, maxc = region.bbox
        indices = np.mgrid[minr:maxr, minc:maxc]
        region_image = image[(indices[0], indices[1])]
    else:
        region_image = region.intensity_image

    _, _, database[f"{tag} Fourier SDI"] = (0, 0, 0)#fourier_transform_analysis(segment_image)

    database[f"{tag} Mean"] = np.mean(region_image)
    database[f"{tag} STD"] = np.std(region_image)
    database[f"{tag} Entropy"] = shannon_entropy(region_image)
    database[f"{tag} Density"] = np.sum(region_image * region.image) / region.area

    glcm = greycomatrix(
        (region_image * region.image * 255.999).astype('uint8'),
        [1, 2], [0, np.pi/4, np.pi/2, np.pi*3/4], 256,
        symmetric=True, normed=True)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0

    greycoprops = greycoprops_edit(glcm)

    database[f"{tag} GLCM Contrast"] = greycoprops['contrast'].mean()
    database[f"{tag} GLCM Homogeneity"] = greycoprops['homogeneity'].mean()
    database[f"{tag} GLCM Energy"] = greycoprops['energy'].mean()
    database[f"{tag} GLCM Entropy"] = greycoprops['entropy'].mean()
    database[f"{tag} GLCM Autocorrelation"] = greycoprops['autocorrelation'].mean()
    database[f"{tag} GLCM Clustering"] = greycoprops['cluster'].mean()
    database[f"{tag} GLCM Mean"] = greycoprops['mean'].mean()
    database[f"{tag} GLCM Covariance"] = greycoprops['covariance'].mean()
    database[f"{tag} GLCM Correlation"] = greycoprops['correlation'].mean()

    return database


def network_metrics(network, network_red, tag=''):
    """Analyse networkx Graph object"""

    database = pd.Series(dtype=object)

    cross_links = np.array([degree[1] for degree in network.degree], dtype=int)
    database[f"{tag} Network Cross-Links"] = (cross_links > 2).sum()

    try:
        database[f"{tag} Network Degree"] = nx.degree_pearson_correlation_coefficient(network, weight='r')**2
    except:
        database[f"{tag} Network Degree"] = None

    try:
        database[f"{tag} Network Eigenvalue"] = np.real(nx.adjacency_spectrum(network_red).max())
    except:
        database[f"{tag} Network Eigenvalue"] = None

    try:
        database[f"{tag} Network Connectivity"] = nx.algebraic_connectivity(network_red, weight='r')
    except:
        database[f"{tag} Network Connectivity"] = None

    return database


def fibre_metrics(tot_fibres):
    """Analysis of list of `Fibre` objects

    Parameters
    ----------
    tot_fibres : list of `<class: Fibre>`
        List of fibre to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from networkx Graph and scikit-image
        regionrrops objects
    """

    database = pd.DataFrame()

    for fibre in tot_fibres:
        fibre_series = fibre.generate_database()
        database = database.append(fibre_series, ignore_index=True)

    return database


def fibre_network_metrics(fibre_networks, image=None, sigma=0.0001):
    """Analysis of list of `FibreNetwork` objects

    Parameters
    ----------
    fibre_networks : list of `<class: FibreNetwork>`
        List of fibre networks to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from networkx Graph and scikit-image
        regionprops objects
    """

    database = pd.DataFrame()

    if image is not None:
        nematic_tensor = form_structure_tensor(image, sigma)
    else:
        nematic_tensor = form_structure_tensor(
            fibre_networks[0].image, sigma)

    for i, fibre_network in enumerate(fibre_networks):
        # if segment.filled_area >= 1E-2 * image_shg.size:

        fibre_network_series = pd.Series(name=i, dtype=object)

        metrics = fibre_network.generate_database()

        nematic_metrics = nematic_tensor_metrics(
            fibre_network.region, nematic_tensor, 'Fibre Network')

        fibre_network_series = pd.concat(
            (fibre_network_series, metrics, nematic_metrics))

        metrics = fibre_metrics(fibre_network.fibres)
        mean_metrics = metrics.mean()

        fibre_network_series['Mean Fibre Waviness'] = mean_metrics['Fibre Waviness']
        fibre_network_series['Mean Fibre Length'] = mean_metrics['Fibre Length']
        fibre_network_series['SHG Fibre Cross-Link Density'] = (
                fibre_network_series['SHG Network Cross-Links'] / len(metrics))

        database = database.append(fibre_network_series, ignore_index=True)

    # fibre_angle_sdi[i] = angle_analysis(fibre_ang, np.ones(fibre_ang.shape))

    return database


def cell_metrics(cells, image=None, sigma=0.0001):
    """Analysis of a list of `Cell` objects

    Parameters
    ----------
    cells : list of `<class: Cell>`
        List of cells to analyse

    Returns
    -------
    database : DataFrame
        Metrics calculated from scikit-image regionprops objects
    """
    database = pd.DataFrame()

    if image is not None:
        nematic_tensor = form_structure_tensor(image, sigma)
    else:
        nematic_tensor = form_structure_tensor(
            cells[0].image, sigma)

    for i, cell in enumerate(cells):

        cell_series = cell.generate_database()

        nematic_metrics = nematic_tensor_metrics(
            cell.region, nematic_tensor, 'Cell')
        cell_series = pd.concat((cell_series, nematic_metrics))

        database = database.append(cell_series, ignore_index=True)

    return database
