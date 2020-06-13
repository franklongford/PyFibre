Running the PyFibre CLI
-----------------------

Once installed, enter the PyFibre deployment environment using::

    python -m ci shell

Calling the executable ``PyFibre`` from the command line will initiate the terminal based version of PyFibre::

	Usage: PyFibre [OPTIONS] [FILE_PATHS]

    Options:
      --version             Show the version and exit.
      --debug               Prints extra debug information in
                            pyfibre.log
      --profile             Run GUI under cProfile, creating .prof and
                            .pstats files in the current directory.
      --ow_metric           Toggles overwrite analytic metrics
      --ow_segment          Toggles overwrite image segmentation
      --ow_network          Toggles overwrite network extraction
      --save_figures        Toggles saving of figures
      --test                Perform run on test image
      --key TEXT            Keywords to filter file names
      --sigma FLOAT         Gaussian smoothing standard deviation
      --alpha FLOAT         Alpha network coefficient
      --database_name TEXT  Output database filename
      --log_name TEXT       Pyfibre log filename
      --help                Show this message and exit.

Multiple file paths can be included as arguments, including directories. The user may also include also use
wildcard assignments, for example::

    PyFibre some-images/ other-images/some-file-path.tif more-images/other-files*

