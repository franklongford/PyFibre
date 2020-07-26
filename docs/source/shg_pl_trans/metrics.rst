Metrics
~~~~~~~

Metrics are grouped together according to which PyFibre objects they are calculated from (fibre network, fibre segment,
cell segment, global image). The 4 database types are:

Fibre Network
^^^^^^^^^^^^^
Contain metrics calculated from each network of fibres extracted by the FIRE algorithm::

    <file-name>_network_metric.h5
    <file-name>_network_metric.xls

.. csv-table:: Fibre Network Metrics
    :header: "Metric", "Description", "Category"
    :widths: 20, 30, 10

    "No. Fibres", "Number of extracted fibres", "Content"
    "Mean Fibre Waviness", "Average fibre waviness (length / displacement)", "Content"
    "Mean Fibre Length", "Average fibre pixel length", "Content"
    "Fibre Angle SDI", "Angle spectrum SDI (mean / max) of fibre angles", "Content"
    "Fibre Network Degree", "Average fibre network number of edges per node", "Network"
    "Fibre Network Eigenvalue", "Max eigenvalue of network adjacency matrix", "Network"
    "Fibre Network Connectivity", "Average fibre network connectivity", "Network"
    "Fibre Network Cross-Link Density", "Average cross-links per fibre", "Network"

Fibre Segment
^^^^^^^^^^^^^
Contain metrics calculated from each fibre segment identified in the segmentation process::

    <file-name>_fibre_metric.h5
    <file-name>_fibre_metric.xls

.. csv-table:: Fibre Segment Metrics
    :header: "Metric", "Description", "Category"
    :widths: 20, 30, 10

    "Fibre Segment Area", "Area of fibre segment (in pixels)", "Shape"
    "Fibre Segment Coverage", "Ratio of fibre segment area / bounding box area", "Content"
    "Fibre Segment Linearity", "Ratio of fibre segment diameter / perimeter ", "Shape"
    "Fibre Segment Eccentricity", "`Eccentricity <https://en.wikipedia.org/wiki/Eccentricity_(mathematics)>`_ shape metric of fibre segment", "Shape"
    "Fibre Segment SHG Angle SDI", "Angle spectrum SDI (mean / max) for all SHG image pixels in fibre segment", "Texture"
    "Fibre Segment SHG Anisotropy", "Anisotropy of mean structure tensor of all SHG image pixels in fibre segment", "Texture"
    "Fibre Segment SHG Local Anisotropy", "Mean anisotropy of structure tensors of all SHG image pixels in fibre segment", "Texture"
    "Fibre Segment SHG Mean", "Mean intensity of fibre segment pixels in SHG image", "Texture"
    "Fibre Segment SHG STD", "Standard deviation of fibre segment pixels in SHG image", "Texture"
    "Fibre Segment SHG Entropy", "Average Shannon entropy of fibre segment pixels in SHG image"

Cell Segment
^^^^^^^^^^^^
Contain metrics calculated from each cell segment identified in the segmentation process::

    <file-name>_cell_metric.h5
    <file-name>_cell_metric.xls

.. csv-table:: Cell Segment Metrics
    :header: "Metric", "Description", "Category"
    :widths: 20, 30, 10

    "No. Cells", "Number of cell segments", Content
    "Cell Segment Area", "Area of cell segment (in pixels)", "Shape"
    "Cell Segment Coverage", "Ratio of cell segment area / bounding box area", "Content"
    "Cell Segment Linearity", "Ratio of cell segment diameter / perimeter ", "Shape"
    "Cell Segment Eccentricity", "`Eccentricity <https://en.wikipedia.org/wiki/Eccentricity_(mathematics)>`_ shape metric of cell segment", "Shape"
    "Cell Segment PL Angle SDI", "Angle spectrum SDI (mean / max) for all PL image pixels in cell segment", "Texture"
    "Cell Segment PL Anisotropy", "Anisotropy of mean structure tensor of all PL image pixels in cell segment", "Texture"
    "Cell Segment PL Local Anisotropy", "Mean anisotropy of structure tensors of all PL image pixels in cell segment", "Texture"
    "Cell Segment PL Mean", "Mean intensity of cell segment pixels in PL image", "Texture"
    "Cell Segment PL STD", "Standard deviation of cell segment pixels in PL image", "Texture"
    "Cell Segment PL Entropy", "Average Shannon entropy of cell segment pixels in SHG image"

Global Image
^^^^^^^^^^^^
Contain metrics calculated from the global image::

    <file-name>_global_metric.h5
    <file-name>_global_metric.xls

Contains averages of all network and segment metrics that are created during the analysis process

Output Files
~~~~~~~~~~~~

PyFibre will create a folder with relevant analysis for each multi image that is loaded into the software. These
contain database files containing all metrics for each identified object in the SHG-PL-Trans images
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

The additional raw data files in the ``data`` directory contain serialised copies of the segment and network
objects used in the analysis::

    file-name_network.pkl

Contains a pickled copy of the NetworkX graph used to represent the fibre networks::

    file-name_fibre_segments.npy
    file-name_cell_segments.npy

Contain stacks of NumPy arrays representing pixel masks of each fibre and cell segment identified in the
segmentation process.

Databases
~~~~~~~~~

Databases are also generated from all SHG-PL-Trans images that are loaded into the ``PyFibre`` software during a
session. The names of these output files can be customised by setting the ``--database_name`` flag in the CLI or
specifying a name in the "Save Database" GUI tool::

    <database_name>_global.h5
    <database_name>_global.xls

Contains a set of global metrics for each image::

    <database_name>_network.h5
    <database_name>_network.xls

Contains metrics for every fibre network in each image::

    <database_name>_fibre.h5
    <database_name>_fibre.xls

Contains metrics for every fibre segment in each image::

    <database_name>_cell.h5
    <database_name>_cell.xls

Contains metrics for every cell segment in each image.
