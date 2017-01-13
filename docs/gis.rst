.. _gis:

Map transformations
===================

Most of the georeferencing machinery for gridded datasets is
handled by the :py:class:`~salem.Grid` class: its capacity to handle
gridded datasets in a painless manner was one of the primary
motivations to develop Salem.

Grids
-----

A point on earth can be defined unambiguously in two ways:

DATUM (lon, lat, datum)
    longitudes and latitudes are angular coordinates of a point on an
    ellipsoid (often called "datum")
PROJ (x, y, projection)
    x (eastings) and y (northings) are cartesian coordinates of a point in a
    map projection (the unit of x, y is usually meter)

Salem adds a third coordinate reference system (**crs**) to this list:

GRID (i, j, Grid)
    on a structured grid, the (x, y) coordinates are distant of a
    constant (dx, dy) step. The (x, y) coordinates are therefore equivalent
    to a new reference frame (i, j) proportional to the projection's (x, y)
    frame.

Transformations between datums and projections is handled by several tools
in the python ecosystem, for example `GDAL`_ or the more lightweight
`pyproj`_, which is the tool used by Salem internally [#]_.

The concept of Grid added by Salem is useful when transforming data between
two structured datasets, or from an unstructured dataset to a structured one.

.. _GDAL: https://pypi.python.org/pypi/GDAL/
.. _pyproj: https://jswhit.github.io/pyproj/


.. [#] Most datasets nowadays are defined in the WGS 84 datum, therefore the
       concepts of datum and projection are often interchangeable:
       (lon, lat) coordinates are equivalent to cartesian (x, y) coordinates
       in the plate carree projection.

A :py:class:`~salem.Grid` is defined by a projection, a reference point in
this projection, a grid spacing and a number of grid points:

.. ipython:: python

    import numpy as np
    import salem
    from salem import wgs84

    grid = salem.Grid(nxny=(3, 2), dxdy=(1, 1), x0y0=(0.5, 0.5), proj=wgs84)
    x, y = grid.xy_coordinates
    x
    y

The default is to define the grids according to the pixels center point:

.. ipython:: python

    smap = salem.Map(grid)
    smap.set_data(np.arange(6).reshape((2, 3)))
    lon, lat = grid.ll_coordinates
    smap.set_points(lon.flatten(), lat.flatten())

    @savefig plot_example_grid.png width=80%
    smap.visualize(addcbar=False)

But with the ``pixel_ref`` keyword you can use another convention. For Salem,
the two conventions are identical:

.. ipython:: python

    grid_c = salem.Grid(nxny=(3, 2), dxdy=(1, 1), x0y0=(0, 0),
                        proj=wgs84, pixel_ref='corner')
    assert grid_c == grid

While it's good to know how grids work, most of the time grids should be
inferred directly from the data files (see also: :ref:`xarray_acc.init`):

.. ipython:: python

    ds = salem.open_xr_dataset(salem.get_demo_file('himalaya.tif'))
    grid = ds.salem.grid
    grid.proj.srs
    grid.extent

Grids come with several convenience functions, for example for transforming
points onto the grid coordinates:

.. ipython:: python

    grid.transform(85, 27, crs=salem.wgs84)

Or for reprojecting structured data as explained below.


Reprojecting data
-----------------

Interpolation
~~~~~~~~~~~~~

The standard way to reproject a gridded dataset into another one is to use the
:py:func:`~salem.DatasetAccessor.transform` method:


.. ipython:: python
   :suppress:

    plt.rcParams['figure.figsize'] = (7, 3)
    f = plt.figure(figsize=(7, 3))

.. ipython:: python

    dse = salem.open_xr_dataset(salem.get_demo_file('era_interim_tibet.nc'))
    t2_era_reproj = ds.salem.transform(dse.t2m.isel(time=0))

    @savefig plot_reproj_grid.png width=80%
    t2_era_reproj.salem.quick_map();

This is the recommended way if the output grid (in this case, a high resolution
lon-lat grid) is of similar or finer resolution than the input grid (in this
case, reanalysis data at 0.75°). As of v0.2, three interpolation methods are
available in Salem: ``nearest`` (default), ``linear``, or ``spline``:


.. ipython:: python

    t2_era_reproj = ds.salem.transform(dse.t2m.isel(time=0), interp='spline')

    @savefig plot_reproj_grid_spline.png width=80%
    t2_era_reproj.salem.quick_map();

Internally, Salem uses `pyproj <https://jswhit.github.io/pyproj/>`__ for the
coordinates transformation and scipy's interpolation methods for the
resampling. Note that reprojecting data can be computationally and
memory expensive: it is generally recommended to reproject your data at the
end of the processing chain if possible.

The :py:func:`~salem.DatasetAccessor.transform` method returns an object of
the same structure as the input. The only differences are the coordinates and
the grid, which are those of the arrival grid:


.. ipython:: python

    dst = ds.salem.transform(dse)
    dst
    dst.salem.grid == ds.salem.grid

Aggregation
~~~~~~~~~~~

If you need to resample higher resolution data onto a coarser grid,
:py:func:`~salem.DatasetAccessor.lookup_transform` may be the way to go. This
method gets its name from the "lookup table" it uses internally to store
the information needed for the resampling: for each
grid point in the coarser dataset, the lookup table stores the coordinates
of the high-resolution grid located below.

The default resampling method is to average all these points:

.. ipython:: python

    dse = dse.salem.subset(corners=((77, 23), (94.5, 32.5)))
    dsl = dse.salem.lookup_transform(ds)
    @savefig plot_lookup_grid.png width=80%
    dsl.data.salem.quick_map(cmap='terrain');

But any aggregation method is available, for example ``np.std``, or ``len`` if
you want to know the number of high resolution pixels found below a coarse
grid point:

.. ipython:: python

    dsl = dse.salem.lookup_transform(ds, method=len)
    @savefig plot_lookup_grid_std.png width=80%
    dsl.data.salem.quick_map();
