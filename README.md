# PyFibre

![Python application](https://github.com/franklongford/PyFibre/workflows/Python%20application/badge.svg?branch=dev)

PyFibre is an open source image analysis toolkit for fibrous tissue that can be run either in a terminal or GUI. It is designed to make the quantification of fibrous tissue automated, standardised and efficient whilst remaining as transparent as possible for non-technical users.
 
![PyFibre logo](pyfibre/gui/images/icon.ico)


## Installation

PyFibre is deployed using Enthought Deployment Manager, 
[EDM](https://www.enthought.com/product/enthought-deployment-manager/). 
Please download and install
the latest version prior to continuing further. 

Once installed, simply create a default environment using the command line:

    edm install --version 3.6 -y click setuptools
    edm shell

Then build the deployment `PyFibre-py36` environment using the following command:

    python -m ci build-env

Afterwards, install a package egg with all binaries using:

    python -m ci install

This will install all required libraries and create the local `PyFibre` and `PyFibre_GUI` binaries.
To make sure the installation has been successful, please run the unittests

    python -m ci test

And create the documentation

    python -m ci docs

## Running the PyFibre GUI

Once installed, enter the deployment environment using

    edm shell -e PyFibre-py36

and call the executable `PyFibre_GUI` from the command line to initiate the graphical user interface.

![GUI](docs/main_view.png)

### File Viewer

PyFibre is set by default to detect Tagged Image Format (tif) files. To load in individual files to analyse, use the filwindow to navigate through your file treethe `Add Files` button and . Alternatively, you can load in all tif files within a single directory by using the `Load Folder` button.

#### Automatic SHG and PL Image Detection

If you have performed multiple imaging techniques on the same region, then PyFibre is able to take advantage of this to provide extra analysis. Currently both Second Harmonic Generation (SHG) and Photoluminescence (PL) imaging techniques are supported. 

Loading images containing the keyword `SHG` or `PL` in the file path (see below) will allow PyFibre automatically match these up based on the `{prefix}`.

	{directory}/{prefix}-{keyword}{suffix}.tif

The files will then appear as:

	{directory}/{prefix}
	
#### File Management

Once loaded, the files are visible in a scrollable list on the left hand side. They can be removed from here at any 
time by highlighting and clicking the `Delete` button, or automatically filtered for keywords using the `Filter` entry form. 

#### Running Analysis

Clicking the `Run` button at the bottom will begin a batch analysis of all the files listed within the box at the 
time of execution. This can be interrupted at any point using the `Stop` button.

### Image Viewer

The image display notebook on the right hand side of the GUI is able to show both the original images as well as results of PyFibre's analysis.

Tab | Description
--- | ---
Loaded Image | Greyscale multi-channel image
Tensor Image | RGB multi-channel image, using hue, saturation and brightness based on pixel structure tensor
Network | Greyscale multi-channel image with overlayed FIRE networks
Fibre | Greyscale multi-channel image with overlayed individual fibres extracted from FIRE networks
Fibre Segment | Greyscale multi-channel image with overlayed segmented regions base on position of FIRE networks
Cell Segment| Greyscale multi-channel image with overlayed segmented regions base on position of cellular regions
Metrics | List of measured properties for multi-channel image

## Running in the Terminal


Calling the executable `PyFibre` from the command line will initiate the terminal based version of PyFibre.

	Usage: PyFibre [OPTIONS] [FILE_PATH]

    Options:
      --version             Show the version and exit.
      --debug               Prints extra debug information in
                            pyfibre.log
      --profile             Run GUI under cProfile, creating .prof and
                            .pstats files in the current directory.
      --ow_metric           Toggles overwrite analytic metrics
      --ow_segment          Toggles overwrite image segmentation
      --ow_network          Toggles overwrite network extraction
      --save_figures        Toggles saving of figures
      --test                Perform run on test image
      --key TEXT            Keywords to filter file names
      --sigma FLOAT         Gaussian smoothing standard deviation
      --alpha FLOAT         Alpha network coefficient
      --database_name TEXT  Output database filename
      --log_name TEXT       Pyfibre log filename
      --help                Show this message and exit.
    

## Metrics

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
