Extending PyFibre
-----------------

PyFibre can be extended using the `Envisage <https://docs.enthought.com/envisage/index.html>`_ plugin framework.
We provide the ``PyFibrePlugin`` class, designed to be used to support additional image formats. Each plugin may
contribute objects to support multiple image formats.

PyFibre Factories
~~~~~~~~~~~~~~~~~

Each plugin contributes a set of ``IMultiImageFactory`` classes that contains:

1. An image file type, provided by a class that fulfils the ``IMultiImage`` interface.
2. A reader for the image file type, provided by a class that fulfils the ``IMultiImageReader`` interface.
3. An analysis script for image file type, provided by a class that fulfils the ``IMultiImageAnalysis`` interface.

In practice, a developer should use the following abstract base classes:

``BaseMultiImage``
^^^^^^^^^^^^^^^^^^

Represents a stack of images that contain each signal for a given position. Provides the ``IMultiImage`` interface
and requires the following methods to be implemented:

.. code-block:: python

    @classmethod
    def verify_stack(cls, image_stack):
        """Perform verification that image_stack is allowed by
        subclass of IMultiImage"""

    def preprocess_images(self):
        """Implement operations that are used to pre-process
        the image_stack before analysis"""

``BaseMultiImageReader``
^^^^^^^^^^^^^^^^^^^^^^^^

Provides the ``IMultiImageReader`` interface, requires the following methods to be implemented

.. code-block:: python

    @abstractmethod
    def collate_files(self, filenames):
        """Returns a dictionary of file sets that can be loaded
        in as an image stack

        Returns
        -------
        image_dict: dict(str, list of str)
            Dictionary containing file references as keys and a list of
            files able to be loaded in as an image stack as values"""

    @abstractmethod
    def create_image_stack(self, filenames):
        """Return a list of numpy arrays suitable for the
        loader's BaseMultiImage type"""

    @abstractmethod
    def load_image(self, filename):
        """Load a single image from a file"""

    @abstractmethod
    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""


``BaseMultiImageAnalyser``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Provides the ``IMultiImageAnalysis`` interface, requires the following methods to be implemented

.. code-block:: python

    @abstractmethod
    def image_analysis(self, *args, **kwargs):
        """Perform analysis on data"""

    @abstractmethod
    def create_metrics(self, *args, **kwargs):
        """Create metrics from multi-image components that can be
        generated upon end of analysis"""

    @abstractmethod
    def create_figures(self, *args, **kwargs):
        """Create figures from multi-image components that can be
        generated upon end of analysis"""

Creating a Plugin
~~~~~~~~~~~~~~~~~

All plugin classes must

- Inherit from ``pyfibre.api.BasePyFibrePlugin``

.. code-block:: python

    from pyfibre.api import BasePyFibrePlugin

    class ExamplePlugin(BasePyFibrePlugin):
    """This is an example of plugin for PyFibre."""

- Implement the methods ``get_name()`` and ``get_version()`` to return appropriate values.

.. code-block:: python

    def get_name(self):
        return "My example plugin"

    def get_version(self):
        return 0

- Implement a method ``get_multi_image_factories()`` returning a list of all contributed classes
  that provide the ``IMultiImageFactory`` interface.

.. code-block:: python

    def get_multi_image_factories(self):
        return [
            ExampleMultiImageFactory
        ]


Install the Plugin
~~~~~~~~~~~~~~~~~~

In order for PyFibre to recognize the plugin, it must be installed as a package in the deployment edm environment, using
the entry point namespace ``pyfibre.plugins``. This can be performed using ``pip`` and an appropriate ``setup.py`` file,
that employs the ``setuptools`` `package <https://setuptools.readthedocs.io/en/latest/setuptools.html>`_.

A basic example ``setup.py`` file is therefore shown below

.. code-block:: python

    from setuptools import setup, find_packages

    setup(
        name="my_example_plugin",
        version=0,
        entry_points={
            "pyfibre.plugins": [
                "my_example = "
                "my_example.example_plugin:ExamplePlugin",
        ]
        },
        # Automatically looks for file directories containing __init__.py files
        # to be included in package
        packages=find_packages(),
    )

Running the following command line instruction from the same directory as ``setup.py`` will then install
the package in the deployed environment

.. code-block:: console

    edm run -e pyfibre-py36 -- pip install -e .
