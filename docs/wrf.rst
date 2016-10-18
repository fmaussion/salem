.. _wrf:

.. currentmodule:: salem

WRF tools
=========

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



Geogrid simulator
-----------------

Salem provides a small tool which comes handy when defining new WRF
domains. It parses the WPS configuration file (`namelist.wps`_) and generates
the grids and maps corresponding to each domain.

.. _namelist.wps: http://www2.mmm.ucar.edu/wrf/OnLineTutorial/Basics/GEOGRID/geogrid_namelist.htm

:py:func:`~geogrid_simulator` will search for the ``&geogrid`` section of the
file and parse it:

.. ipython:: python

    fpath = get_demo_file('namelist_mercator.wps')
    with open(fpath, 'r') as f:  # this is just to show the file
        print(f.read())

    from salem import geogrid_simulator
    g, maps = geogrid_simulator(fpath)

    maps[0].set_rgb(natural_earth='lr')  # add a background image
    @savefig plot_geo_simu.png width=100%
    maps[0].visualize(title='Domains 1 to 4')
