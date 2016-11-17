.. _wrf:

.. currentmodule:: salem

WRF tools
=========

Let's open a `WRF model`_ output file with xarray first:

.. _WRF Model: http://www2.mmm.ucar.edu/wrf/users/


.. ipython:: python
   :suppress:

    plt.rcParams['figure.figsize'] = (6, 4)
    f = plt.figure(figsize=(6, 4))

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

    ds.attrs = {'note': 'Global attrs removed.',
                'pyproj_srs': ds.attrs['pyproj_srs']}

This parser greatly simplifies the file structure:

.. ipython:: python

    ds

Note that some dimensions / coordinates have been renamed, new variables have
been defined, and the staggered dimensions have disappeared.


Diagnostic variables
--------------------

Salem adds a layer between xarray and the underlying NetCDF file. This layer
computes new variables "on the fly" or, in the case of staggered variables,
"unstaggers" them:

.. ipython:: python

    ds.U

This computation is done only on demand (just like a normal
NetCDF variable), this new layer is therefore relatively cheap.

In addition to unstaggering, Salem adds a number of "diagnostic" variables
to the dataset. Some are convenient (like ``T2C``, temperature in Celsius
instead of Kelvins), but others are more complex (e.g. ``SLP`` for sea-level
pressure, or ``PRCP`` which computes step-wize total precipitation out of the
accumulated fields). For a list of diagnostic variables (and TODOs!), refer to
:issue:`18`.

.. ipython:: python

    @savefig plot_wrf_diag.png width=80%
    ds.PRCP.isel(time=-1).salem.quick_map(cmap='Purples', vmax=5)


Vertical interpolation
----------------------

The WRF vertical coordinates are eta-levels, which is not a very practical
coordinate for analysis or plotting. With the functions
:py:func:`~DatasetAccessor.wrf_zlevel` and
:py:func:`~DatasetAccessor.wrf_plevel` it is possible to interpolate the 3d
data at either altitude or pressure levels:

.. ipython:: python

    ws_h = ds.isel(time=1).salem.wrf_zlevel('WS', levels=10000.)
    @savefig plot_wrf_zinterp.png width=80%
    ws_h.salem.quick_map();

Note that currently, the interpolation is quite slow, see :issue:`25`. It's
probably wise to do it on single time slices or aggregated data, rather than
huge data cubes.


Open multiple files
-------------------

It is possible to open multiple WRF files at different time steps with
:py:func:`~salem.open_mf_wrf_dataset`, which works like xarray's
`open_mfdataset`_ . The only drawback of having multiple WRF time slices is
that "de-accumulated" variables such as ``PRCP`` won't be available.
For this purpose, you might want to use the :py:func:`~DataArrayAccessor.deacc`
method.

.. _open_mfdataset: http://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html#xarray.open_mfdataset


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
