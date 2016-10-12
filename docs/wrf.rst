.. _wrf:

WRF data
========

Let's open a `WRF model`_ output file with xarray first:

.. _WRF Model: http://www2.mmm.ucar.edu/wrf/users/

.. ipython:: python

    import xarray as xr
    from salem.utils import get_demo_file
    ds = xr.open_dataset(get_demo_file('wrfout_d01.nc'))

.. ipython:: python
   :suppress:

    ds.attrs = {'note': 'Global attrs removed.'}

WRF files are a bit messy:

.. ipython:: python

    ds

WRF files aren't exactly CF compliant: you'll need a special parser for the
timestamp, the coordinate names are a bit exotic and do not correspond
to the dimension names, they contain so-called `staggered`_ variables
(and their correponding coordinates), etc.

.. _staggered: https://en.wikipedia.org/wiki/Arakawa_grids


Salem defines a special parser for these files:


.. ipython:: python

    import salem
    ds = salem.open_wrf_dataset(get_demo_file('wrfout_d01.nc'))

.. ipython:: python
   :suppress:

    ds.attrs = {'note': 'Global attrs removed.'}

This parser greatly simplifies the file structure:

.. ipython:: python

    ds
