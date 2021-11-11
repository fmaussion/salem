.. _installing:

Installation
============

Salem is a pure python package, but it has several non-trivial dependencies.

Required dependencies
---------------------

- Python 2.7 or 3+ (Py3 `recommended <https://python3statement.github.io/>`__)
- `numpy <http://www.numpy.org/>`__ (of course)
- `scipy <http://scipy.org/>`__: for its interpolation tools, among other things
- `pyproj <https://jswhit.github.io/pyproj/>`__: for map projections transformations
- `netCDF4 <https://github.com/Unidata/netcdf4-python>`__: to read most geoscientific files
- `pandas <http://pandas.pydata.org/>`__: working with labeled data
- `xarray <https://jswhit.github.io/pyproj/>`__ (0.8 or later): pandas in N-dimensions
- `joblib <https://pythonhosted.org/joblib/>`__: for it's `Memory`_ class
- `six <https://pythonhosted.org/six//>`__: for Py2 compatibility

.. _Memory: https://pythonhosted.org/joblib/memory.html

Optional dependencies
---------------------

For vector and raster operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `rasterio <https://mapbox.github.io/rasterio/>`__: for geotiff files
- `shapely <https://pypi.python.org/pypi/Shapely>`__: geometric objects
- `geopandas <http://geopandas.org/>`__: geospatial data with pandas

(rasterio and geopandas both require GDAL)

For plotting
~~~~~~~~~~~~

- `matplotlib <http://matplotlib.org/>`__: required for :ref:`plotting`
- `pillow <http://pillow.readthedocs.io>`__: required for salem.Map
- `scikit-image <https://scikit-image.org>`__: required for salem.Map
- `motionless <https://github.com/ryancox/motionless/>`__: for google static maps


Instructions
------------

You can install Salem via `conda-forge`_::

    conda config --add channels conda-forge
    conda install salem

or pip::

    pip install salem

The very best (unique?) way to install Salem without too much hassle is to
install its dependencies with `conda`_ and `conda-forge`_::

    conda config --add channels conda-forge
    conda install <package-name>

Currently, Salem can only be installed via pip::

    pip install salem

If you want to install the latest master::

    pip install git+https://github.com/fmaussion/salem.git

.. _conda-forge: http://conda-forge.github.io

.. warning::

    At the first import, Salem will create a hidden directory called
    ``.salem_cache`` in your home folder. It will be used to download Salem's
    demo files and standard shapefiles. This directory is also used by
    joblib to store the result of slow operations such as reading and
    transforming shapefiles, or downloading google maps from the internet. The
    cache should not become too large, but if it does: simply delete it.
