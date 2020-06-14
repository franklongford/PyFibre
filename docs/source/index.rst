Welcome to PyFibre documentation!
====================================

.. image:: https://github.com/franklongford/PyFibre/workflows/Python%20application/badge.svg?branch=dev
    :target: https://github.com/franklongford/PyFibre/tree/dev
    :alt: Python application

.. image:: https://readthedocs.org/projects/pyfibre-docs/badge/?version=latest
    :target: https://pyfibre-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

PyFibre (Python Fibrous Image Analysis Toolkit) is an open source, extensible image analysis toolkit for fibrous tissue
that can be run either in a terminal or GUI. It is designed to be as automated as possible so that it can easily be applied to
multiple image sets in order to generate databases of properties for further large-scale analysis.

PyFibre has been designed to work with the Tagged Image File Format (TIFF), and is shipped with a plugin that provides
automated analysis routines for a multi-channel image format containing Second Harmonic Generation (SHG) and
Phospholuminescene (PL) signals.

User Manual
===========

.. toctree::
   :maxdepth: 2

   Installation instructions <installation>
   Running PyFibre <pyfibre_apps>
   Core Features <core/core_features>
   Extending PyFibre <plugins>
   SHG-PL Analysis <shg_pl_analysis>


API Reference
=============

.. toctree::
   :maxdepth: 2

   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`