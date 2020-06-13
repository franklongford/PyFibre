Automatic SHG and PL Image Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have performed multiple imaging techniques on the same region, then PyFibre is able to take advantage of this to
provide extra analysis. Currently both Second Harmonic Generation (SHG) and Photoluminescence (PL) imaging techniques
are supported.

Loading images containing the keyword ``SHG`` or ``PL`` (non case sensitive) in the file path (see below) will allow
PyFibre automatically match these up based on the ``{prefix}``::

    {directory}/{prefix}-{keyword}{suffix}.tif

The files will then appear as::

    {directory}/{prefix}


