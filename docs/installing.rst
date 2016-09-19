.. _installing:

Installation
============

Salem is a pure python package, but it has several non-trivial dependencies.

Required dependencies
---------------------

- Python 2.7 or 3+ (Py3 `recommended <https://python3statement.github.io/>`__)
- `numpy <http://www.numpy.org/>`__ (of course)
- `scipy <http://scipy.org/>`__: for its interpolation tools, among other things
- `pyproj <https://jswhit.github.io/pyproj/>`__: for map projections
- `netCDF4 <https://github.com/Unidata/netcdf4-python>`__: to read most geoscientific files
- `joblib <https://pythonhosted.org/joblib/>`__: for it's `Memory`_ class
- `six <https://pythonhosted.org/six//>`__: for Py2 compatibility

.. _Memory: https://pythonhosted.org/joblib/memory.html

Optional dependencies
---------------------

Because not using them is a bad idea
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `pandas <http://pandas.pydata.org/>`__: working with labeled data
- `xarray <https://jswhit.github.io/pyproj/>`__: pandas in N-dimensions

For vector and raster operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(rasterio and geopandas require GDAL)

- `rasterio <https://mapbox.github.io/rasterio//>`__: for geotiff files
- `shapely <https://pypi.python.org/pypi/Shapely>`__: geometric objects
- `geopandas <http://geopandas.org/>`__: geospatial data with pandas

For plotting
~~~~~~~~~~~~

- `matplotlib <http://matplotlib.org/>`__: required for :ref:`plotting`
- `pillow <http://pillow.readthedocs.io/en/latest/installation.html>`__: required for maps
- `descartes <https://pypi.python.org/pypi/descartes/>`__: for paths and patches on maps
- `motionless <https://github.com/ryancox/motionless/>`__: for google static maps


Instructions
------------

The very best (unique?) way to install Salem without too much hassle is to
install its dependencies with `conda`_ and `conda-forge`_::

    conda config --add channels conda-forge
    conda install <package-name>

For the moment, Salem can only be installed with pip::

    pip install git+https://github.com/fmaussion/salem.git

.. _conda: http://conda.pydata.org/docs/intro.html
.. _conda-forge: http://conda-forge.github.io

.. warning::

    At the first import, Salem will create a hidden directory called ``.salem_cache``
    in your home folder. It will be used to download Salem's
    demo files and standard shapefiles. This directory is also used by
    joblib to store the result of some slow operations such as reading and
    transforming shapefiles, or downloading google maps from the internet. The
    cache should not become too large, but if it does: simply delete it.
