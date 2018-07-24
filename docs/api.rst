#############
API reference
#############

Here we will add the documentation for selected modules.

.. currentmodule:: salem

Grid
====

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Grid

Grid attributes
---------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Grid.proj
    Grid.nx
    Grid.ny
    Grid.dx
    Grid.dy
    Grid.x0
    Grid.y0
    Grid.origin
    Grid.pixel_ref
    Grid.x_coord
    Grid.y_coord
    Grid.xy_coordinates
    Grid.ll_coordinates
    Grid.xstagg_xy_coordinates
    Grid.ystagg_xy_coordinates
    Grid.xstagg_ll_coordinates
    Grid.ystagg_ll_coordinates
    Grid.center_grid
    Grid.corner_grid
    Grid.extent

Grid methods
------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Grid.extent_in_crs
    Grid.ij_to_crs
    Grid.almost_equal
    Grid.region_of_interest
    Grid.regrid
    Grid.transform
    Grid.map_gridded_data
    Grid.grid_lookup
    Grid.lookup_transform
    Grid.to_dict
    Grid.from_dict
    Grid.to_json
    Grid.from_json
    Grid.to_dataset
    Grid.to_geometry


Georeferencing utils
====================

.. autosummary::
    :toctree: generated/
    :nosignatures:

    check_crs
    proj_is_same
    proj_to_cartopy
    transform_proj
    transform_geometry
    transform_geopandas
    mercator_grid
    grid_from_dataset
    reduce


Graphics
========

.. autosummary::
    :toctree: generated/
    :nosignatures:

    DataLevels
    Map
    get_cmap

Map & DataLevels methods
------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    DataLevels.append_colorbar
    DataLevels.colorbarbase
    DataLevels.set_cmap
    DataLevels.set_data
    DataLevels.set_plot_params
    DataLevels.set_extend
    DataLevels.set_levels
    DataLevels.set_nlevels
    DataLevels.set_vmax
    DataLevels.set_vmin
    DataLevels.visualize
    DataLevels.plot

Map methods
-----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Map.set_data
    Map.set_contour
    Map.set_contourf
    Map.set_geometry
    Map.set_lonlat_contours
    Map.set_points
    Map.set_rgb
    Map.set_shapefile
    Map.set_text
    Map.set_topography
    Map.set_scale_bar
    Map.transform
    Map.visualize
    Map.plot

Input/output
============

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_demo_file
    read_shapefile
    read_shapefile_to_grid


xarray
======

.. autosummary::
    :toctree: generated/
    :nosignatures:

    open_xr_dataset
    open_metum_dataset
    open_wrf_dataset
    open_mf_wrf_dataset


xarray accessors
----------------

Salem adds `accessors`_ to xarray objects. They can be accessed via the
``.salem`` attribute and add the following methods:

.. _accessors: http://xarray.pydata.org/en/stable/internals.html#extending-xarray


DataArray
^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    DataArrayAccessor.subset
    DataArrayAccessor.roi
    DataArrayAccessor.transform
    DataArrayAccessor.lookup_transform
    DataArrayAccessor.cartopy
    DataArrayAccessor.get_map
    DataArrayAccessor.quick_map
    DataArrayAccessor.interpz
    DataArrayAccessor.deacc


Dataset
^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    DatasetAccessor.subset
    DatasetAccessor.roi
    DatasetAccessor.transform
    DatasetAccessor.lookup_transform
    DatasetAccessor.transform_and_add
    DatasetAccessor.cartopy
    DatasetAccessor.get_map
    DatasetAccessor.quick_map
    DatasetAccessor.wrf_zlevel
    DatasetAccessor.wrf_plevel


Old-style datasets
==================

Old-style Datasets (prior to xarray), kept for backwards compatibility
reasons and because they are quite lightweight. They might be replaced by
xarray's datasets one day.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    GeoDataset
    GeoTiff
    GeoNetcdf
    WRF
    GoogleCenterMap
    GoogleVisibleMap


Other
=====

.. autosummary::
    :toctree: generated/
    :nosignatures:

    geogrid_simulator
