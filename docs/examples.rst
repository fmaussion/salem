.. _examples:

Examples
========

Subsetting and selecting data
-----------------------------

Let's open a `WRF model`_ output file:

.. _WRF Model: http://www2.mmm.ucar.edu/wrf/users/

.. ipython:: python

    import salem
    from salem.utils import get_demo_file
    ds = salem.open_xr_dataset(get_demo_file('wrfout_d01.nc'))

Let's take a time slice of the variable ``T2`` for a start:

.. ipython:: python

    t2 = ds.T2.isel(Time=2)

    @savefig plot_wrf_t2.png width=80%
    t2.salem.quick_map()

Although we are on a Lambert Conformal projection, it's possible to subset
the file using longitudes and latitudes:

.. ipython:: python

    t2_sub = t2.salem.subset(corners=((77., 20.), (97., 35.)), crs=salem.wgs84)

    @savefig plot_wrf_t2_corner_sub.png width=80%
    t2_sub.salem.quick_map()

It's also possible to use geometries or shapefiles to subset your data:


.. ipython:: python

    shdf = salem.read_shapefile(get_demo_file('world_borders.shp'))
    shdf = shdf.loc[shdf['CNTRY_NAME'].isin(['Nepal', 'Bhutan'])]  # GeoPandas' GeoDataFrame
    t2_sub = t2_sub.salem.subset(shape=shdf, margin=2)  # add 2 grid points

    @savefig plot_wrf_t2_country_sub.png width=80%
    t2_sub.salem.quick_map()

Based on the same principle, one can mask out the useless grid points:

.. ipython:: python

    t2_roi = t2_sub.salem.roi(shape=shdf)

    @savefig plot_wrf_t2_roi.png width=80%
    t2_roi.salem.quick_map()


Plotting
--------

Maps can be pimped with topographical shading, points of interest,
and more:

.. ipython:: python

    smap = t2_roi.salem.get_map(data=t2_roi-273.15, cmap='RdYlBu_r', vmin=-14, vmax=18)
    _ = smap.set_topography(get_demo_file('himalaya.tif'))
    smap.set_shapefile(shape=shdf, color='grey', linewidth=3)
    smap.set_points(91.1, 29.6)
    smap.set_text(91.2, 29.7, 'Lhasa', fontsize=17)

    @savefig plot_wrf_t2_topo.png width=80%
    smap.visualize()

Maps are persistent, which is useful when you have many plots to do. Plotting
further data on them is possible, as long
as the geolocalisation information is shipped with the data (in that case,
the DataArray's attributes are lost in the conversion from Kelvins to degrees
Celsius so we have to set it explicitly):


.. ipython:: python

    smap.set_data(ds.T2.isel(Time=1)-273.15, crs=ds.salem.grid)

    @savefig plot_wrf_t2_transform.png width=80%
    smap.visualize(title='2m temp - large domain', cbar_title='C')


Reprojecting data
-----------------

Salem can also transform data from one grid to another:

.. ipython:: python

        dse = salem.open_xr_dataset(get_demo_file('era_interim_tibet.nc'))
        t2_era_reproj = ds.salem.transform(dse.t2m)
        assert t2_era_reproj.salem.grid == ds.salem.grid
        @savefig plot_era_repr_nn.png width=80%
        t2_era_reproj.isel(time=0).salem.quick_map()



.. ipython:: python

        t2_era_reproj = ds.salem.transform(dse.t2m, interp='spline')
        @savefig plot_era_repr_spline.png width=80%
        t2_era_reproj.isel(time=0).salem.quick_map()
