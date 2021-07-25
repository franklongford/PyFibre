Change Log
==========

Release 2.1.0
-------------

Date:
~~~~~
25/07/2021

Authors:
~~~~~~~~
Frank Longford

Description:
~~~~~~~~~~~~
Includes new features to GUI image display that enables

Features
^^^^^^^^
- Lazy loading of images in each tab improves performance (#85)
- Include brightness slider in GUI (#88)

Fixes
^^^^^
- Fix to normalise intensity levels of saved image PNGs (#87)
- Fix to networkx random state and handling of weights as a list in nanmean routine (#82)

Maintenance
^^^^^^^^^^^
- Bump ETS packages to support Traits Futures version 2.0 (#82)
- Include explict dependency on decorator package to fix issue with networkx


Release 2.0.4
-------------

Date:
~~~~~
12/04/2021

Authors:
~~~~~~~~
Frank Longford

Description:
~~~~~~~~~~~~
Fixes to minor release 2.0.3


Fixes
^^^^^
- Use raw intensity values in SHG-PL-Trans image segments (#80)

Release 2.0.3
-------------

Date:
~~~~~
11/04/2021

Authors:
~~~~~~~~
Frank Longford

Description:
~~~~~~~~~~~~
Fixes to RTD builds and SHG-PL-Trans routines

Maint
^^^^^
- Update requirements file for ReadTheDocs builds (#74)

Fixes
^^^^^
- FIX for SHG and PL Analysis (#75)
- Include clipping in network building and segmentation for SHG-PL-Trans images (#77)

Release 2.0.2
-------------

Date:
~~~~~
20/09/2020

Authors:
~~~~~~~~
Frank Longford

Description:
~~~~~~~~~~~~
Some refactoring to plugins modules and amendments to SHG-PL-Trans metrics


Maint
^^^^^
- Includes CI testing on Windows (#53)
- SHG-PL-Trans plugin now contained in pyfibre.addons module (#65)

Fixes
^^^^^
- Rename Linearity to Circularity metric (#66)
- Redefine coverage metrics for global data sets (#66)


Release 2.0.1
-------------

Date:
~~~~~
19/07/2020

Authors:
~~~~~~~~
Frank Longford

Description:
~~~~~~~~~~~~
Minor patch fix for database generation

Fixes
^^^^^
- Fix broken database generation in PyFibre GUI
- Segment texture and structure tensor metrics are calculated using pixels in segment masks


Release 2.0.0
-------------

Date:
~~~~~
08/07/2020

Authors:
~~~~~~~~
Frank Longford

Description:
~~~~~~~~~~~~
Major release using ETS for front and backend. Complete rewrite from previous Tk GUI.
Deployed using EDM with framework that allows extension points.