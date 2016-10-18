.. _plotting:

.. currentmodule:: salem



.. ipython:: python
   :suppress:

    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (7, 2)
    import numpy as np

Plotting
========

Color handling with DataLevels
------------------------------

:py:class:`~DataLevels` is the base class for handling colors. It is there to
ensure that there will never be a mismatch between data and assigned colors.

.. ipython:: python

    from salem import DataLevels
    a = [-1., 0., 1.1, 1.9, 9.]

    dl = DataLevels(a)
    @savefig datalevels_01.png width=70%
    dl.visualize(orientation='horizontal', add_values=True)

Discrete levels
~~~~~~~~~~~~~~~

.. ipython:: python

    dl.set_plot_params(nlevels=11)
    @savefig datalevels_02.png width=70%
    dl.visualize(orientation='horizontal', add_values=True)

vmin, vmax
~~~~~~~~~~

.. ipython:: python

    dl.set_plot_params(nlevels=9, vmax=3)
    @savefig datalevels_03.png width=70%
    dl.visualize(orientation='horizontal', add_values=True)

Out-of-bounds data
~~~~~~~~~~~~~~~~~~

.. ipython:: python

    dl.set_plot_params(levels=[0, 1, 2, 3])
    @savefig datalevels_04.png width=70%
    dl.visualize(orientation='horizontal', add_values=True)

Note that if the bounds are not exceeded, the colorbar extensions are gone:

.. ipython:: python

    dl.set_data([0., 0.2, 0.4, 0.6, 0.8])
    @savefig datalevels_05.png width=70%
    dl.visualize(orientation='horizontal', add_values=True)

This might be undesirable, so you can set a keyword to force out-of-bounds
levels:

.. ipython:: python

    dl.set_plot_params(levels=[0, 1, 2, 3], extend='both')
    @savefig datalevels_06.png width=70%
    dl.visualize(orientation='horizontal', add_values=True)


Using DataLevels with matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Here with the example of a scatterplot:

.. ipython:: python

    x, y = np.random.randn(1000), np.random.randn(1000)
    z = x**2 + y**2

    dl = DataLevels(z, cmap='RdYlBu_r', levels=np.arange(6))
    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax.scatter(x, y, color=dl.to_rgb(), s=64);
    cbar = dl.append_colorbar(ax, "right")  # DataLevel draws the colorbar
    @savefig datalevels_07.png width=70%
    plt.show()


Maps
----

:py:class:`~Map` is a sublass of :py:class:`~DataLevels`, but adds the
georeferencing aspects to the plot. A Map is initalised with a
:py:class:`~Grid`:


.. ipython:: python

    from salem import mercator_grid, Map, get_demo_file
    grid = mercator_grid(center_ll=(10.76, 46.79), extent=(2e6, 1e6))
    smap = Map(grid)
    @savefig map_central_europe.png width=100%
    smap.visualize(addcbar=False)


.. ipython:: python
   :suppress:

    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (6, 4)

Add topographical shading to a map
----------------------------------

You can add topographical shading to a map with DEM files:

.. ipython:: python

    from salem import mercator_grid, Map, get_demo_file
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

