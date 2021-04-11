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
3. An analysis script for the image file type, provided by a class that fulfils the ``IMultiImageAnalysis`` interface.
4. A parser for the image file type, provided by a class that fulfils the ``IFileParser`` interface.
5. A set of view objects for the image file type, provided by a class that fulfils the ``IMultiImageViewer`` interface.


In practice, a developer should use the following abstract base classes:

``BaseMultiImage``
^^^^^^^^^^^^^^^^^^

Represents a stack of images that contain each signal for a given position. Provides the ``IMultiImage`` interface
and requires the following methods to be implemented:

.. code-block:: python

    @classmethod
    def verify_stack(cls, image_stack):
        """Perform verification that image_stack is allowed by this
        subclass of BaseMultiImage. Returns either True or False, depending
        on whether the image_stack argument is correctly formatted.
        """

    def preprocess_images(self):
        """Implement operations that are used to pre-process
        the image_stack before analysis.
        """


``BaseFileParser``
^^^^^^^^^^^^^^^^^^

Collates sets of image files together into ``FileSet`` objects, which can be used to
create ``BaseMultiImage`` objects by loading single or multiple files.

.. code-block:: python

    def get_file_sets(self, filenames):
        """From a given list of file names, returns a dictionary where each entry
        represents the files required to create an instance of a multi image.
        Each key will be passed on as the name of the multi image, used during
        further PyFibre operations. Each value could be passed in as the
        `filenames` argument to the class `create_image_stack` method.

        Returns
        -------
        file_sets: list of FileSet
            List containing FileSet objects that hold a collection of files to
            be loaded in as a single image
        """


``BaseMultiImageReader``
^^^^^^^^^^^^^^^^^^^^^^^^

Is able to take a ``FileSet`` object and use it to load an image file or a set of
images files as a ``BaseMultiImage`` object. Provides the ``IMultiImageReader``
interface, and requires the following methods to be implemented

.. code-block:: python

    def get_multi_image_class(self):
        """Returns class of IMultiImage that will be loaded."""

    def get_supported_file_sets(self):
        """Returns class of IFileSets that will be supported."""

    def get_filenames(self, file_set):
        """From a collection of files in a FileSet, yield each file that
        should be used
        """

    def load_image(self, filename):
        """Load a single image from a file"""

    def can_load(self, filename):
        """Perform check to see whether file is formatted
        correctly to be loaded"""


``BaseMultiImageAnalyser``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Contains analysis scripts for a specific ``BaseMultiImage`` class. Provides the ``IMultiImageAnalysis`` interface,
and requires the following methods to be implemented

.. code-block:: python

    def image_analysis(self, *args, **kwargs):
        """Perform analysis on data"""

    def create_metrics(self, *args, **kwargs):
        """Create metrics from multi-image components that can be
        generated upon end of analysis"""

    def create_figures(self, *args, **kwargs):
        """Create figures from multi-image components that can be
        generated upon end of analysis"""


``BaseFileParser``
^^^^^^^^^^^^^^^^^^

Contains routines to parse a single file or collection of files as an ``IMultiImage`` class. These
are packaged together into ``FileSet`` objects.


.. code-block:: python

    def get_file_sets(self, filenames):
        """From a given list of file names, returns a dictionary where each entry
        represents the files required to create an instance of a multi image."""


``BaseMultiImageViewer``
^^^^^^^^^^^^^^^^^^^^^^^^

Contributes UI objects that can be used to display ``BaseMultiImage`` objects and the results of their analysis
in the PyFibre GUI.

.. code-block:: python

    def create_display_tabs(self):
        """Returns a list of objects providing the IDisplayTab
        interface"""

    def update_display_tabs(self):
        """Updates each display tab when called"""


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
