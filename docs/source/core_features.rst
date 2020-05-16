Automatic SHG and PL Image Detection
------------------------------------

If you have performed multiple imaging techniques on the same region, then PyFibre is able to take advantage of this to provide extra analysis. Currently both Second Harmonic Generation (SHG) and Photoluminescence (PL) imaging techniques are supported.

Loading images containing the keyword `SHG` or `PL` in the file path (see below) will allow PyFibre automatically match these up based on the `{prefix}`.

	{directory}/{prefix}-{keyword}{suffix}.tif

The files will then appear as:

	{directory}/{prefix}

Metrics
-------

PyFibre calculates properties for the global images and each segmentd region. The resultant databases for each section are then labelled:

	{directory}/data/{prefix}.h5 = global image output (also in .xls format)
	{directory}/data/{prefix}_fibre.h5 = fibre segmented image output (also in .xls format)
	{directory}/data/{prefix}_cell.h5 = cell segmented image output (also in .xls format)

The database metrics have the following columns:

Metric | Description | Category
--- | --- | ---
No. Fibres | Number of extracted fibres | Content
Fibre Waviness | Average fibre waviness (length / displacement) | Content
Fibre Lengths | Average fibre pixel length | Content
Fibre Angles | Average fibre angle | Content
Fibre Segment Coverage | Ratio of image containing fibres | Content
Fibre Segment Area | Area of image (in pixels) containing fibres | Shape
Fibre Segment Linearity | Average ratio of diameter to perimeter for fibre segments | Shape
Fibre Segment Eccentricity | Average eccentricity metric of fibre segments | Shape
Fibre Segment SHG Angle SDI | Angle spectrum SDI (mean / max) for all SHG image pixels in fibre regions| Texture
Fibre Segment SHG Anisotropy | Anisotropy of structure tensor for all SHG image pixels in fibre regions | Texture
Fibre Segment SHG Pixel Anisotropy | Mean anisotropy of structure tensor for all SHG image pixels in fibre regions | Texture
Fibre Segment SHG Mean | Mean pixel intensity of SHG image in fibre segment | Texture
Fibre Segment SHG STD | Standard deviation of pixel intensity of SHG image in fibre segment | Texture
Fibre Segment SHG Entropy | Average Shannon entropy of pixel intensities of SHG image in fibre segment | Texture
Fibre Network Degree | Average fibre network number of edges per node | Network
Fibre Network Eigenvalue | Max eigenvalue of network adjacency matrix| Network
Fibre Network Connectivity | Average fibre network connectivity | Network
Fibre Network Cross-Link Density | Average cross-links per fibre | Content
No. Cells | Number of cell segments | Content
Cell Segment Coverage | Ratio of image containing fibres/cells | Content
Cell Segment Area | Area of image (in pixels) containing cells | Shape
Cell Segment Linearity | Average ratio of diameter to perimeter for cell segments | Shape
Cell Segment Eccentricity | Average eccentricity metric of cell segments | Shape
Cell Segment PL Angle SDI | Angle spectrum SDI (mean / max) for all PL image pixels in cell regions| Texture
Cell Segment PL Anisotropy | Anisotropy of structure tensor for all PL image pixels in cell regions | Texture
Cell Segment PL Pixel Anisotropy | Mean anisotropy of structure tensor for all PL image pixels in cell regions | Texture
Cell Segment PL Intensity Mean | Mean pixel intensity of PL image in cell segment | Texture
Cell Segment PL Intensity STD | Standard deviation of pixel intensity of PL image in cell segment | Texture
Cell Segment PL Intensity Entropy | Average Shannon entropy of pixel intensities of PL image in cell segment | Texture
