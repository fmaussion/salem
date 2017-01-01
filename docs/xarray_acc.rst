.. _xarray_acc:

Xarray accessors
================

One of the main purposes of Salem is to add georeferencing tools to
`xarray`_ 's data structures. These tools can be accessed via a `special`_
``.salem`` attribute, available for both `xarray.DataArray`_ and
`xarray.Dataset`_ objects after a simple ``import salem`` in your code.

.. _xarray: http://xarray.pydata.org/
.. _special: http://xarray.pydata.org/en/stable/internals.html#extending-xarray
.. _xarray.DataArray: http://xarray.pydata.org/en/stable/data-structures.html#dataarray
.. _xarray.Dataset: http://xarray.pydata.org/en/stable/data-structures.html#dataset

.. _xarray_acc.init:

Initializing the accessor
-------------------------

Automated projection parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Salem will try to understand in which projection your Dataset or DataArray is
defined. For example, with a platte carree (or lon/lat) projection, Salem will
know what to do based on the coordinates' names:

.. ipython:: python

    import numpy as np
    import xarray as xr
    import salem

    da = xr.DataArray(np.arange(20).reshape(4, 5), dims=['lat', 'lon'],
                      coords={'lat':np.linspace(0, 30, 4),
                              'lon':np.linspace(-20, 20, 5)})

    da.salem  # the accessor is an object

    @savefig plot_xarray_simple.png width=80%
    da.salem.quick_map();

While the above should work with a majority of climate datasets (such as
atmospheric reanalyses or GCM output), certain NetCDF files will have a more
exotic map projection requiring a dedicated parsing. There are `conventions`_
to formalise these things in the NetCDF data model, but Salem doesn't
understand them yet (my impression is that they aren't widely used anyways).

Currently, Salem can deal with:

- platte carree (or lon/lat) projections
- WRF projections (see :ref:`wrf`)
- for geotiff files only: any projection that `rasterio`_ can understand
- **virually any projection provided explicitly by the user**

.. _conventions: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch05s06.html
.. _rasterio: https://mapbox.github.io/rasterio/

The logic for deciding upon the projection of a Dataset or DataArray is
located in :py:func:`~salem.grid_from_dataset`. If the automated parsing
doesn't work, the ``salem`` accessor won't work either. In that case, you'll
have to provide your own :ref:`xarray_acc.custom` to the data.

From geotiffs
~~~~~~~~~~~~~

Salem uses `rasterio`_ to open and parse geotiff files:

.. ipython:: python
   :suppress:

    plt.rcParams['figure.figsize'] = (7, 3)
    f = plt.figure(figsize=(7, 3))

.. ipython:: python

    fpath = salem.get_demo_file('himalaya.tif')
    ds = salem.open_xr_dataset(fpath)
    hmap = ds.salem.get_map(cmap='topo')
    hmap.set_data(ds['data'])

    @savefig plot_xarray_geotiff.png width=80%
    hmap.visualize();

.. _xarray_acc.custom:

Custom projection
~~~~~~~~~~~~~~~~~

Alternatively, Salem will understand any projection supported by  `pyproj`_.
The proj info has to be provided as attribute:

.. _pyproj: https://jswhit.github.io/pyproj/

.. ipython:: python

    dutm = xr.DataArray(np.arange(20).reshape(4, 5), dims=['y', 'x'],
                        coords={'y': np.arange(3, 7)*2e5,
                                'x': np.arange(1, 6)*2e5})
    psrs = 'epsg:32630'  # http://spatialreference.org/ref/epsg/wgs-84-utm-zone-30n/
    dutm.attrs['pyproj_srs'] = psrs


    @savefig plot_xarray_utm.png width=80%
    dutm.salem.quick_map(interp='linear');


Using the accessor
------------------

The accessor's methods are available trough the ``.salem`` attribute.

Keeping track of attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some datasets carry their georeferencing information in global attributes (WRF
model output files for example). This makes it possible for Salem to
determine the data's map projection. From the variables alone,
however, this is not possible. This is the reason why it is recommended to
use the :py:func:`~salem.open_xr_dataset` and
:py:func:`~salem.open_wrf_dataset` function, which add
an attribute to the variables automatically:

.. ipython:: python

    dsw = salem.open_xr_dataset(salem.get_demo_file('wrfout_d01.nc'))
    dsw.T2.pyproj_srs

Unfortunately, the DataArray attributes are lost when doing operations on them.
It is the task of the user to keep track of this attribute:

.. ipython:: python

    dsw.T2.mean(dim='Time', keep_attrs=True).salem  # triggers an error without keep_attrs


Reprojecting data
~~~~~~~~~~~~~~~~~

.. ipython:: python
   :suppress:

    plt.rcParams['figure.figsize'] = (7, 3)
    f = plt.figure(figsize=(7, 3))

You can reproject a Dataset onto another one with the
:py:func:`~salem.DatasetAccessor.transform` function:

.. ipython:: python

    dse = salem.open_xr_dataset(salem.get_demo_file('era_interim_tibet.nc'))
    dsr = ds.salem.transform(dse)
    dsr
    @savefig plot_xarray_transfo.png width=80%
    dsr.t2m.mean(dim='time').salem.quick_map();

Currently, salem implements, the neirest neighbor (default), linear, and spline
interpolation methods:

.. ipython:: python

    dsr = ds.salem.transform(dse, interp='spline')
    @savefig plot_xarray_transfo_spline.png width=80%
    dsr.t2m.mean(dim='time').salem.quick_map();


The accessor's map transformation machinery is handled by the
:py:class:`~salem.Grid` class internally. See :ref:`gis` for more information.


Subsetting data
~~~~~~~~~~~~~~~

The :py:func:`~salem.DatasetAccessor.subset` function allows you to subset
your datasets according to (georeferenced) vector or raster data, for example
based on `shapely <https://pypi.python.org/pypi/Shapely>`__ geometries
(e.g. polygons), other grids, or shapefiles:

.. ipython:: python

    shdf = salem.read_shapefile(salem.get_demo_file('world_borders.shp'))
    shdf = shdf.loc[shdf['CNTRY_NAME'] == 'Nepal']  # remove other countries
    dsr = dsr.salem.subset(shape=shdf, margin=10)
    @savefig plot_xarray_subset_out.png width=80%
    dsr.t2m.mean(dim='time').salem.quick_map();


Regions of interest
~~~~~~~~~~~~~~~~~~~

While subsetting selects the optimal rectangle over your region of interest,
sometimes you also want to maskout unrelevant data, too. This is done with the
:py:func:`~salem.DatasetAccessor.roi` tool:

.. ipython:: python

    dsr = dsr.salem.roi(shape=shdf)
    @savefig plot_xarray_roi_out.png width=80%
    dsr.t2m.mean(dim='time').salem.quick_map();
