.. _installing:

Installation
============

Required dependencies
---------------------

- Python 2.7 or 3+ (we `recommend <https://python3statement.github.io/>`__ to use python 3)
- `numpy <http://www.numpy.org/>`__
- `scipy <http://scipy.org/>`__
- `pyproj <https://jswhit.github.io/pyproj/>`__
- `netCDF4 <https://github.com/Unidata/netcdf4-python>`__

Optional dependencies
---------------------

Because not using them is a bad idea
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `pandas <http://pandas.pydata.org/>`__: working with labeled data
- `xarray <https://jswhit.github.io/pyproj/>`__: pandas in N-dimensions

For netCDF and IO
~~~~~~~~~~~~~~~~~

- `scipy <http://scipy.org/>`__: used as a fallback for reading/writing netCDF3
- `pydap <http://www.pydap.org/>`__: used as a fallback for accessing OPeNDAP
- `h5netcdf <https://github.com/shoyer/h5netcdf>`__: an alternative library for
  reading and writing netCDF4 files that does not use the netCDF-C libraries
- `pynio <https://www.pyngl.ucar.edu/Nio.shtml>`__: for reading GRIB and other
  geoscience specific file formats

For shapefiles and vector data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `shapely <https://pypi.python.org/pypi/Shapely>`__: geometric objects
- `geopandas <http://geopandas.org/>`__: geospatial data with pandas

For plotting
~~~~~~~~~~~~

- `matplotlib <http://matplotlib.org/>`__: required for :ref:`plotting`
- `seaborn <https://stanford.edu/~mwaskom/software/seaborn/>`__: for nicer plots


Instructions
------------

