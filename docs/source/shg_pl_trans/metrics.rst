Metrics
~~~~~~~

PyFibre calculates properties for the global images and each segmented region. The resultant databases for
each section are then labelled::

``{directory}/data/{prefix}_global.h5`` = global image output (also in .xls format)
``{directory}/data/{prefix}_fibre.h5`` = fibre segmented image output (also in .xls format)
``{directory}/data/{prefix}_network.h5`` = fibre networks output (also in .xls format)
``{directory}/data/{prefix}_cell.h5`` = cell segmented image output (also in .xls format)

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