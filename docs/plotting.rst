.. _plotting:

.. currentmodule:: salem

Graphics
========

Two options are offered to you when plotting geolocalised data on maps:
you can use `cartopy`_ , or you can use salem's :py:class:`~Map` object.

.. _cartopy: http://scitools.org.uk/cartopy/docs/latest/index.html


Plotting with cartopy
---------------------

Plotting on maps using xarray and cartopy is extremely
`convenient <http://xarray.pydata.org/en/stable/plotting.html#maps>`_.

With salem you can keep your usual plotting workflow, even with more exotic
map projections:

.. ipython:: python

    import matplotlib.pyplot as plt
    import cartopy
    from salem import open_wrf_dataset, get_demo_file
    ds = open_wrf_dataset(get_demo_file('wrfout_d01.nc'))

    ax = plt.axes(projection=cartopy.crs.Orthographic(70, 30))
    ax.set_global();
    ds.T2C.isel(time=1).plot.contourf(ax=ax, transform=ds.salem.cartopy());
    @savefig cartopy_orthographic.png width=80%
    ax.coastlines();


You can also use the salem accessor to initialise the plot's map projection:

.. ipython:: python

    proj = ds.salem.cartopy()
    ax = plt.axes(projection=proj)
    ax.coastlines();
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':');
    @savefig cartopy_base.png width=80%
    ax.set_extent(ds.salem.grid.extent, crs=proj);


Plotting with salem
-------------------

Salem comes with a homegrown plotting tool. It is less flexible than
cartopy, but it was created to overcome some of cartopy's limitations (e.g.
the impossibility to add tick labels to lambert conformal maps), and to make
nice looking regional maps:

.. ipython:: python

    @savefig salem_quickmap.png width=80%
    ds.T2C.isel(time=1).salem.quick_map()

Salem maps are different from cartopy's in that they don't change the
matplotlib axes' projection. The map background is always going to be a
call to `imshow()`_, with an image size decided at instanciation:

.. ipython:: python
   :suppress:

    plt.rcParams['figure.figsize'] = (7, 3)
    f = plt.figure(figsize=(7, 3))

.. ipython:: python

    from salem import mercator_grid, Map, open_xr_dataset

    grid = mercator_grid(center_ll=(10.76, 46.79), extent=(9e5, 4e5))
    grid.nx, grid.ny  # size of the input grid
    smap = Map(grid, nx=500)
    smap.grid.nx, smap.grid.ny  # size of the "image", and thus of the axes

    @savefig map_central_europe.png width=100%
    smap.visualize(addcbar=False)

The map has it's own grid, wich is used internally in order to transform
the data that has to be plotted on it:

.. ipython:: python

    ds = open_xr_dataset(get_demo_file('histalp_avg_1961-1990.nc'))
    smap.set_data(ds.prcp)  # histalp is a lon/lat dataset
    @savefig map_histalp.png width=100%
    smap.visualize()

Refer to :ref:`recipes` for more examples on how to use salem's maps.

.. _imshow(): http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow