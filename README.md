# PyFibre
### Python image analysis library for fibrous tissue

PyFibre is an open source toolkit that can be run either in a terminal or GUI

Installation
------

PyFibre requires a local distributions of python >= 3.6 and pip >= 9.0 in order to run. Either anaconda or miniconda distributions are recommended, as well as the use of a virtual environment, since creating local binaries may require root access.

Once downloaded, run `make install` in the PyFibre directory to install all required libraries and create the `PyFibre` and `PyFibre_GUI` binaries.

Running the PyFibre GUI
----

Once installed, calling the executable `PyFibre_GUI` will initiate a graphical user interface.

PyFibre is set by default to detect Tagged Image Format (tif) files. To load in individual files to analyse, use the `Select File` button and use the pop up window to navigate through your file tree. Alternatively, you can load in all tif files within a single directory by using the `Select Folder` button.

Once loaded, the files are visible in a scrollable list on the left hand side. They can be removed from here at any time by highlighting and clicking the `Delete` button. However, clicking the `GO` button at the bottom will begin a batch analysis of all the files listed within the box at the time of execution.


Running in the Terminal
----



Parameters

	Image Anisotropy : Anisotropy of nematic tensor for whole image
	Pixel Anisotropy : Average anisotropy of nematic tensor for each pixel in the image
	Network Clustering : Clustering coefficient of extracted fibre network
