SHG-PL-Trans Images
-------------------

The PyFibre runner is designed to load in any files that are recognisable by the software and perform a supported
analysis routine. It is expected that such an analysis will generate a set of metrics that can be stored in a one or
more database files. As standard, PyFibre supports read and analysis operations of a particular type of TIFF image by
default (SHG-PL-Trans), which represents a data recorded by SHG and PL spectroscopy techniques.

.. toctree::
   :maxdepth: 1

   File Reader <shg_pl_trans/reader>
   Segmentation Algorithms <shg_pl_trans/segmentation>
   Database Metrics <shg_pl_trans/metrics>
