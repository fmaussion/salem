.. -*- rst -*- -*- restructuredtext -*-
.. This file should be written using restructured text conventions

=====
Salem
=====

.. image:: https://travis-ci.org/fmaussion/salem.svg?branch=master
    :target: https://travis-ci.org/fmaussion/salem

.. image:: https://coveralls.io/repos/fmaussion/salem/badge.svg?branch=master&service=github
  :target: https://coveralls.io/github/fmaussion/salem?branch=master


Salem is a `cat <https://drive.google.com/file/d/0B-0AsTwFw61uSE0zaktOOVN5X1E/view?usp
=sharing>`_. Salem is also a small library to do some geoscientific data
handling. Together with `cleo <https://github.com/fmaussion/cleo>`_, they
provide a framework to work, analyse, and plot climate and geoscientific data.

Bigger (and better) projects are found out there (e.g. Iris): Salem basically
reinvents the wheel, but in the way I want the wheel to be. Salem is quite
young but well tested, and might be useful to a few: if you are using the
`WRF <http://www.wrf-model.org>`_ model for example, or if you don't want to
bother about some aspects of georeferencing of your netCDF files. Another
reason might be that you are a user of my IDL library (WAVE) and want to try
out Python for a change.

Installation
------------

Salem relies on several libraries (six, numpy, scipy, pyproj, pandas, joblib
and matplotlib). To get the full functionality you will also need xray,
rasterio, pandas, geopandas and shapely.

After installing those, a simple *pip install* should suffice::

    $ pip install git+https://github.com/fmaussion/salem.git#egg=salem

This will build the latest repository version. Since Salem is growing fast,
the API is most likely to change in the future.


Classes
-------

**Grid**
    Handling of structured (gridded) map projections

**GeoDataset**
    Handling of gridded data files (subsetting, regions of interest,
    timeseries)

**GeoDataset/GeoTiff**
    Handling of georeferenced TIFF files

**GeoDataset/GeoNetcdf**
    Handling of gridded netCDF files (they should be approx. CF compliant or
    should be implemented in Salem, such as WRF)

**GeoDataset/GeoNetcdf/WRF**
    Handling of WRF output files (including diagnostic variables such as
    SLP, TK, etc.)

**GeoDataset/GoogleCenterMap**
    Download and georeferencing of the Google Static "Center Map" API

**GeoDataset/GoogleVisibleMap**
    Download and georeferencing of the Google Static "Visible Map" API


Getting started with Salem
--------------------------

Coming soon...


About
-----

:License:
    GNU GPLv3

:Author:
    Fabien Maussion - fabien.maussion@uibk.ac.at
