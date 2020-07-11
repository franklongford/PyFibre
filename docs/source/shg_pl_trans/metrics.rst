Metrics
~~~~~~~

PyFibre will create a folder with relevant analysis for each multi image that is loaded into the software. These
contain data base files containing all metrics for each identified object in the SHG-PL-Trans images
in both ``hdf5`` and ``xls`` formatting. PNG images displaying the analysis are also created if the
``--save_figures`` option is selected::

    file-name-shg.tif
    file-name-pl.tif
    file-name-pyfibre-analysis/
    │
    ├── data
    │    │
    │    ├── file-name_global_metric.h5
    │    ├── file-name_global_metric.xls
    │    ├── file-name_cell_metric.h5
    │    ├── file-name_cell_metric.xls
    │    ├── file-name_fibre_metric.h5
    │    ├── file-name_fibre_metric.xls
    │    ├── file-name_network_metric.h5
    │    ├── file-name_network_metric.xls
    │    ├── file-name_network.pkl
    │    ├── file-name_fibre_segments.npy
    │    └── file-name_cell_segments.npy
    │
    │
    └── fig
         ├── file-name_cell_seg.png
         ├── file-name_fibre_seg.png
         ├── file-name_fibre.png
         ├── file-name_network.png
         ├── file-name_tensor.png
         ├── file-name_PL.png
         ├── file-name_SHG.png
         └── file-name_trans.png


The database metrics have the following columns:

.. csv-table:: SHG-PL-Trans Image Metrics
    :header: "Metric", "Description", "Category"
    :widths: 20, 30, 10

    "No. Fibres", "Number of extracted fibres", "Content"
    "Fibre Waviness", "Average fibre waviness (length / displacement)", "Content"
    "Fibre Lengths", "Average fibre pixel length", "Content"
    "Fibre Angles", "Average fibre angle", "Content"
    "Fibre Segment Coverage", "Ratio of image containing fibres", "Content"
    "Fibre Segment Area", "Area of image (in pixels) containing fibres", "Shape"
    "Fibre Segment Linearity", "Average ratio of diameter to perimeter for fibre segments", "Shape"
    "Fibre Segment Eccentricity", "Average eccentricity metric of fibre segments", "Shape"
    "Fibre Segment SHG Angle SDI", "Angle spectrum SDI (mean / max) for all SHG image pixels in fibre regions", "Texture"
    "Fibre Segment SHG Anisotropy", "Anisotropy of structure tensor for all SHG image pixels in fibre regions", "Texture"
    "Fibre Segment SHG Pixel Anisotropy", "Mean anisotropy of structure tensor for all SHG image pixels in fibre regions", "Texture"
    "Fibre Segment SHG Mean", "Mean pixel intensity of SHG image in fibre segment", "Texture"
    "Fibre Segment SHG STD", "Standard deviation of pixel intensity of SHG image in fibre segment", "Texture"
    "Fibre Segment SHG Entropy", "Average Shannon entropy of pixel intensities of SHG image in fibre segment", "Texture"
    "Fibre Network Degree", "Average fibre network number of edges per node", "Network"
    "Fibre Network Eigenvalue", "Max eigenvalue of network adjacency matrix", "Network"
    "Fibre Network Connectivity", "Average fibre network connectivity", "Network"
    "Fibre Network Cross-Link Density", "Average cross-links per fibre", "Content"
    "No. Cells", "Number of cell segments", Content
    "Cell Segment Coverage", "Ratio of image containing fibres/cells", "Content"
    "Cell Segment Area", "Area of image (in pixels) containing cells", "Shape"
    "Cell Segment Linearity", "Average ratio of diameter to perimeter for cell segments", "Shape"
    "Cell Segment Eccentricity", "Average eccentricity metric of cell segments", "Shape"
    "Cell Segment PL Angle SDI", "Angle spectrum SDI (mean / max) for all PL image pixels in cell regions", "Texture"
    "Cell Segment PL Anisotropy", "Anisotropy of structure tensor for all PL image pixels in cell regions", "Texture"
    "Cell Segment PL Pixel Anisotropy", "Mean anisotropy of structure tensor for all PL image pixels in cell regions", "Texture"
    "Cell Segment PL Intensity Mean", "Mean pixel intensity of PL image in cell segment", "Texture"
    "Cell Segment PL Intensity STD", "Standard deviation of pixel intensity of PL image in cell segment", Texture
    "Cell Segment PL Intensity Entropy", "Average Shannon entropy of pixel intensities of PL image in cell segment", "Texture"
