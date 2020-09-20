Change Log
==========

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