Installation
------------

PyFibre is deployed using Enthought Deployment Manager,
`EDM <https://www.enthought.com/product/enthought-deployment-manager/>`_.
Please download and install the latest version prior to continuing further.

Once installed, simply create a "bootstrap" environment using the command line::

    edm install -e bootstrap --version 3.6 -y click setuptools
    edm shell -e bootstrap

To begin with, either clone or download the latest release of Pyfibre (currently 2.0.0) and change working
directory into the repository::

    git clone --branch '2.0.0' --depth 1 https://github.com/franklongford/PyFibre.git
    cd PyFibre

Then build the deployment ``pyfibre-py36`` environment using the following command::

    python -m ci build-env

Afterwards, install a package egg with all binaries using::

    python -m ci install

This will install all required libraries and create the local ``PyFibre`` and ``PyFibre_GUI`` entry points inside the
deployment environment. To make sure the installation has been successful, please enter the deployment environment
and run the integration test provided::

    python -m ci shell
    PyFibre --test
