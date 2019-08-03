# PyFibre

PyFibre is an open source image analysis toolkit for fibrous tissue that can be run either in a terminal or GUI. It is designed to make the quantification of fibrous tissue automated, standardised and efficient whilst remaining as transparent as possible for non-technical users.
 
![PyFibre logo](pyfibre/gui/images/icon.ico)


## Installation

PyFibre requires a local distributions of `python >= 3.6` and `pip >= 9.0` in order to run. 
Either [Enthought](https://www.enthought.com/product/enthought-python-distribution/), [anaconda](https://www.anaconda.com/download/) or [miniconda](https://conda.io/miniconda.html) distributions are recommended.

The use of a package manager, such as [edm](https://www.enthought.com/product/enthought-deployment-manager/) or [conda](https://conda.io/docs/), is optional, but also recommended.

#### EDM Installation (recommended)

A light-weight installation can be performed using the Enthought Deployment Manager (EDM). After downloading [edm](https://www.enthought.com/product/enthought-deployment-manager/), simply create a default environment using:

    edm install --version 3.6 -y click setuptools
    edm shell

Then build the `PyFibre` environment using the following command:

    python -m ci build-env

Afterwards, activate the PyFibre environment and install a package egg with all binaries using:

    edm shell -e PyFibre
    python -m ci install

This will install all required libraries and create the local `PyFibre` and `PyFibre_GUI` binaries.
To make sure the installation has been sucessful, please run the unittests

    python -m ci test

#### Conda Installation

If using anaconda or miniconda python distribution, this can be easily initiated by creating a default environment:

    conda create -n setup python=3.6 -y click setuptools
    source activate setup

Then build the `PyFibre` environment using same command as before but with the `--conda` flag:

    python -m ci build-env --conda

Afterwards, activate the PyFibre environment and install a package egg with all binaries using:

    source activate PyFibre
    python -m ci install --conda

Then run `make` in a cloned or forked PyFibre directory to install the local binaries.
Note: you may need to edit the `PYTHON` and `PIP` variables in the `Makefile`.

## Running the PyFibre GUI

Once installed, calling the executable `PyFibre_GUI` from the command line will initiate a graphical user interface.

![GUI](pyfibre/gui/images/label.png)

### File Viewer

PyFibre is set by default to detect Tagged Image Format (tif) files. To load in individual files to analyse, use the `Load Files` button and use the pop up window to navigate through your file tree. Alternatively, you can load in all tif files within a single directory by using the `Load Folder` button.

#### Automatic SHG and PL Image Detection

If you have performed multiple imaging techniques on the same region, then PyFibre is able to take advantage of this to provide extra analysis. Currently both Second Harmonic Generation (SHG) and Photoluminescence (PL) imaging techniques are supported. 

Loading images containing the keyword `SHG` or `PL` in the file path (see below) will allow PyFibre automatically match these up based on the `{prefix}`.

	{directory}/{prefix}-{keyword}{suffix}.tif

The files will then appear as:

	{directory}/{prefix}
	
#### File Management

Once loaded, the files are visible in a scrollable list on the left hand side. They can be removed from here at any time by highlighting and clicking the `Delete` button, or automatically filtered for keywords using the `Filter` entry form. 

#### Running Analysis

Clicking the `GO` button at the bottom will begin a batch analysis of all the files listed within the box at the time of execution. This can be interrupted at any point using the `STOP` button.

### Image Viewer

The image display notebook on the right hand side of the GUI is able to show both the original images as well as results of PyFibre's analysis.

Tab | Description
--- | ---
SHG Image | Greyscale SHG image
PL Image | Greyscale PL image
Tensor Image | RGB image, using hue, saturation and brightness based on pixel structure tensor
Network | Greyscale SHG image with overlayed FIRE networks
Network Segment | Greyscale SHG image with overlayed segmented regions base on position of FIRE networks
Fibre | Greyscale SHG image with overlayed individual fibres extracted from FIRE networks
Cell Segment| Greyscale PL image with overlayed segmented regions base on position of cellular regions
Metrics | List of measured properties for SHG and PL images
Log | System log

## Running in the Terminal


Calling the executable `PyFibre` from the command line will initiate the terminal based version of PyFibre.

	PyFibre [-h] [--name [NAME]] [--dir [DIR]] [--key [KEY]] 
			[--ow_network] [--ow_segment] [--ow_metric] 
			[--ow_figure] [--save_db [SAVE_DB]] 
			[--threads [THREADS]]

	--name [NAME]        Tif file names to load
	--dir [DIR]          Directories to load tif files
	--key [KEY]          Keywords to filter file names
	--ow_network         Toggles overwrite network extraction
	--ow_segment         Toggles overwrite segmentation
	--ow_metric          Toggles overwrite analytic metrics
	--ow_figure          Toggles overwrite figures
	--save_db [SAVE_DB]  Output database filename
	--threads [THREADS]  Number of threads per processor
	

## Metrics

PyFibre calculates properties for the global images and each segmentd region. The resultant databases for each section are then labelled:

	{directory}/data/{prefix}_global_metric.pkl = global image output (also in .xls format)
	{directory}/data/{prefix}_fibre_metric.pkl = fibre segmented image output (also in .xls format)
	{directory}/data/{prefix}_cell_metric.pkl = cell segmented image output (also in .xls format)

Each database has the following columns:

Metric | Description | Category
--- | --- | ---
No. Fibres | Number of extracted fibres | Network
SHG Angle SDI | Angle spectrum SDI (mean / max) for all SHG image pixels| Texture
SHG Anisotropy | Anisotropy of structure tensor for SHG total image/segment | Texture
SHG Pixel Anisotropy | Mean anisotropy of structure tensor for all SHG image pixels | Texture
SHG Intensity Mean | Mean pixel intensity of SHG total image/segment | Texture
SHG Intensity STD | Standard deviation of pixel intensity of SHG total image/segment | Texture
SHG Intensity Entropy | Average Shannon entropy of pixel intensities of SHG total image/segment | Texture
Fibre GLCM Contrast | GLCM angle-averaged contrast of fibre segment| Texture
Fibre GLCM Homogeneity | GLCM angle-averaged homogeneity of fibre segment| Texture
Fibre GLCM Dissimilarity | GLCM angle-averaged dissimilarity of fibre segment| Texture
Fibre GLCM Correlation | GLCM angle-averaged correlation of fibre segment| Texture
Fibre GLCM Energy | GLCM angle-averaged energy of fibre segment| Texture
Fibre GLCM IDM | GLCM angle-averaged inverse difference moment of fibre segment| Texture
Fibre GLCM Variance | GLCM angle-averaged variance of fibre segment | Texture
Fibre GLCM Cluster | GLCM angle-averaged clustering tendency of fibre segment | Texture
Fibre GLCM Entropy | GLCM angle-averaged entropy of fibre segment | Texture
Fibre Area | Average number of pixels covered by fibres in segment | Content
Fibre Coverage | Ratio of fibre segment covered by fibres  | Content
Fibre Linearity | Average fibre segment linearity | Shape
Fibre Eccentricity | Average fibre segment eccentricity | Shape
Fibre Density | Average image fibre segment density | Texture
Fibre Hu Moment 1 | Average fibre segment Hu moment 1 | Shape
Fibre Hu Moment 2 | Average fibre segment Hu moment 2 | Shape
Network Degree | Average fibre network number of edges per node | Network
Network Eigenvalue | Max eigenvalue of network adjacency matrix| Network
Network Connectivity | Average fibre network connectivity | Network
Fibre Waviness | Average fibre waviness (length / displacement) | Content
Fibre Lengths | Average fibre pixel length | Content
Fibre Cross-Link Density | Average cross-links per fibre | Content
No. Cells | Number of cell segments | Content
Cell Area | Average number of pixels covered by cells | Content
Cell Linearity | Average cell segment linearity | Shape 
Cell Coverage | Ratio of image covered by cell segments | Content
Cell Eccentricity | Average cell segment eccentricity | Shape
Cell Density | Average image cell density | Texture
PL Intensity Mean | Mean pixel intensity of PL total image/segment | Texture
PL Intensity STD | Standard deviation of pixel intensity of PL total image/segment | Texture
PL Intensity Entropy | Average Shannon entropy of pixel intensities of PL total image/segment | Texture
Cell GLCM Contrast | GLCM angle-averaged contrast of cell segment | Texture
Cell GLCM Homogeneity | GLCM angle-averaged homogeneity of cell segment | Texture
Cell GLCM Dissimilarity | GLCM angle-averaged dissimilarity of cell segment | Texture
Cell GLCM Correlation | GLCM angle-averaged correlation of cell segment | Texture
Cell GLCM Energy | GLCM angle-averaged energy of cell segment | Texture
Cell GLCM IDM | GLCM angle-averaged inverse difference moment of cell segment | Texture
Cell GLCM Variance | GLCM angle-averaged variance of cell segment | Texture
Cell GLCM Cluster | GLCM angle-averaged clustering tendency of cell segment | Texture
Cell GLCM Entropy | GLCM angle-averaged entropy  of cell segment | Texture
Cell Hu Moment 1 | Average cell segment Hu moment 1 | Shape
Cell Hu Moment 2 | Average cell segment Hu moment 2 | Shape
