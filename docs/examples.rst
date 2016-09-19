
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

WRF files are not trivial. The projection is hidden somewhere
in the attributes, there are many dimensions (more on these later). Let's
take a time slice of the variable ``T2`` for a start:

.. ipython:: python

    t2 = ds.T2.isel(time=2)
    @savefig plot_wrf_t2.png width=5in
    t2.salem.plot_on_map()

Although we are on a Lambert Conformal projection, it's possible to subset
the file using longitudes and latitudes:

.. ipython:: python

    t2_sub = t2.salem.subset(corners=((77., 20.), (97., 35.)), crs=salem.wgs84)
    @savefig plot_wrf_t2_corner_sub.png width=5in
    t2_sub.salem.plot_on_map()

It's also possible to use geometries or shapefiles to subset your data:


.. ipython:: python

    shdf = salem.read_shapefile(get_demo_file('world_borders.shp'))
    shdf = shdf.loc[shdf['CNTRY_NAME'].isin(['Nepal', 'Bhutan'])]  # GeoPandas' GeoDataFrame
    t2_sub = t2_sub.salem.subset(shape=shdf, margin=5)  # add 5 grid points
    @savefig plot_wrf_t2_country_sub.png width=5in
    t2_sub.salem.plot_on_map()

Based on the same principle, one can mask out the useless grid points:

.. ipython:: python

    t2_roi = t2_sub.salem.roi(shape=shdf)
    @savefig plot_wrf_t2_roi.png width=5in
    smap = t2_roi.salem.plot_on_map()

Maps can be pimped with topographical shading, points of interest,
and more:

.. ipython:: python

    smap.set_topography(get_demo_file('himalaya.tif'))
    smap.set_points(91.1, 29.6)
    smap.set_text(91.2, 29.7, 'Lhasa', fontsize=17)
    @savefig plot_wrf_t2_topo.png width=5in
    smap.visualize()



