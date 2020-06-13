PyFibre
=======

.. image:: https://github.com/franklongford/PyFibre/workflows/Python%20application/badge.svg?branch=dev
    :target: https://github.com/franklongford/PyFibre/tree/dev
    :alt: Python application

.. image:: https://readthedocs.org/projects/pyfibre-docs/badge/?version=latest
    :target: https://pyfibre-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


PyFibre is an open source image analysis toolkit for fibrous tissue that can be run either in a terminal or GUI.
It is designed to make the quantification of fibrous tissue automated, standardised and efficient whilst remaining as
transparent as possible for non-technical users.

Full documentation can be found at the `ReadTheDocs <https://pyfibre-docs.readthedocs.io/en/latest/>`_ page

Installation
------------

PyFibre is deployed using Enthought Deployment Manager, `EDM <https://www.enthought.com/product/enthought-deployment-manager/>`_.
Please download and install the latest version prior to continuing further.

Once installed, simply create a "bootstrap" environment using the command line::

    edm install -e bootstrap --version 3.6 -y click setuptools
    edm shell -e bootstrap

Then build the deployment `pyfibre-py36` environment using the following command::

    python -m ci build-env

Afterwards, install a package egg with all binaries using::

    python -m ci install

This will install all required libraries and create the local `PyFibre` and `PyFibre_GUI` binaries.
To make sure the installation has been successful, please run the unit tests::

    python -m ci test

