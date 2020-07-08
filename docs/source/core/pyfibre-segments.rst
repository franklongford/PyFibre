PyFibre Segments
----------------

PyFibre's segmentation analysis routines aim identify localised regions within each image. Metric calculations
can then be performed on the pixels in each region, as well as the global image. The shape and size of these
segments may also provide insightful data.

Each segment is built on top of the SciKits Image ``RegionProperties`` object.