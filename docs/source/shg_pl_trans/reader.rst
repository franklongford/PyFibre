Automatic SHG and PL Image Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently we support analysis of TIFF files for both Second Harmonic Generation (SHG) and Photoluminescence (PL)
imaging techniques. PyFibre is able to recognise and load in the format of these images if they contain an appropriate
file labelling scheme.

Images containing (non case sensitive) keywords in their file name (see below) will allow PyFibre automatically
match these up based on the ``{prefix}``::

    {directory}/{prefix}-{keyword}{suffix}.tif

The files will then appear as::

    {directory}/{prefix}

It is expected that files containing the ``{keyword}`` ``SHG`` will represent 1 greyscale image (the second harmonic signal),
possibly recorded as a stack.

Files containing the ``{keyword}`` ``PL`` should represent 2 greyscale images (the phospholuminescene signal and its
transmission record), possibly both recorded as separate stacks.

Files containing the combined ``SHG-PL`` ``{keyword}`` will represent 3 greyscale images (SHG, PL and transmission record), with
each possibly recorded as separate stacks.

The file reader requires certain metadata to be present in the TIFF files in order to interpret
which channel corresponds to each signal. Currently only TIFF files created by Olympus FluoView and ImageJ programs
are supported.

Finally, the ``{suffix}`` fragment will be inspected for a reference to the number of accumulations used to record the image.
If detected, the intensity values of each channel in the image will be normalized by this number upon loading.

Example
-------

The following example file name, representing an SHG grayscale image recorded from 6 accumulations, will be segmented accordingly:

``31226-19-4-800nm-3d-medir02-shg-acc6.tif``

``{prefix}``: ``31226-19-4-800nm-3d-medir02``

``{keyword}``: ``shg``

``{suffix}``: ``acc6``
