Running PyFibre
---------------

The ``PyFibreRunner`` is the workhorse of the program, and is used to both load and analyse all input images to either
PyFibre applications.

It iterates through all loaded ``FileSet`` objects, and will perform all analysis scripts that are contributed by
any available contributed ``IMultiImageAnalyser`` class. The runner is called upon invocation of the ``PyFibre`` CLI, or
when selected the "Run" button in the GUI.

.. toctree::
   :maxdepth: 1

   PyFibre CLI <apps/pyfibre_cli>
   PyFibre GUI <apps/pyfibre_gui>