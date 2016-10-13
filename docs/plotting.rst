.. _plotting:

Plotting
========


Add topographical shading to a map
----------------------------------

You can add topographical shading to a map with DEM files:

.. ipython:: python

    from salem import mercator_grid, Map, get_demo_file
    grid = mercator_grid(center_ll=(10.76, 46.79), extent=(20000, 16000))
    smap = Map(grid, countries=False)
    smap.set_topography(get_demo_file('hef_srtm.tif'));
    @savefig topo_shading_simple.png width=100%
    smap.visualize(addcbar=False, title='Topographical shading')

Note that you can also use the topography data to make a colourful plot:

.. ipython:: python

    z = smap.set_topography(get_demo_file('hef_srtm.tif'))
    smap.set_data(z)
    smap.set_cmap('topo')
    @savefig topo_shading_color.png width=100%
    smap.visualize(title='Topography', cbar_title='m a.s.l.')

