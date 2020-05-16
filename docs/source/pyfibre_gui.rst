Running the PyFibre GUI
-----------------------

Once installed, enter the deployment environment using

    edm shell -e PyFibre-py36

and call the executable `PyFibre_GUI` from the command line to initiate the graphical user interface.

![GUI](docs/source/_images/main_view.png)


File Viewer
-----------

PyFibre is set by default to detect Tagged Image Format (tif) files. To load in individual files to analyse, use the filwindow to navigate through your file treethe `Add Files` button and . Alternatively, you can load in all tif files within a single directory by using the `Load Folder` button.

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
