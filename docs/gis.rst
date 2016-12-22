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

    grid = salem.Grid(nxny=(3, 2), dxdy=(1, 1), ll_corner=(0.5, 0.5), proj=wgs84)
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

    grid_c = salem.Grid(nxny=(3, 2), dxdy=(1, 1), ll_corner=(0, 0),
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

TODO: add more :py:func:`~salem.DatasetAccessor.transform` and
:py:func:`~salem.DatasetAccessor.lookup_transform` examples.

.. ipython:: python

    dse = salem.open_xr_dataset(salem.get_demo_file('era_interim_tibet.nc'))
    t2_era_reproj = ds.salem.transform(dse.t2m.isel(time=0))

    @savefig plot_reproj_grid.png width=80%
    t2_era_reproj.salem.quick_map()
