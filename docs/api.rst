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

Grid methods
------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Grid.extent_in_crs
    Grid.ij_to_crs
    Grid.map_gridded_data
    Grid.region_of_interest
    Grid.regrid
    Grid.transform


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

Input/output
============

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_demo_file
    read_shapefile
    read_shapefile_to_grid


Xarray
======

.. autosummary::
    :toctree: generated/
    :nosignatures:

    open_xr_dataset
    open_wrf_dataset
    DataArrayAccessor
    DatasetAccessor


Xarray accessors
----------------

Salem adds `accessors`_ to xarray objects. They can be accessed via the
``.salem`` attribute and add the following methods (DataArray and Dataset
methods are almost equivalent):

.. _accessors: http://xarray.pydata.org/en/stable/internals.html#extending-xarray

.. autosummary::
    :toctree: generated/
    :nosignatures:

    DatasetAccessor.cartopy
    DatasetAccessor.get_map
    DatasetAccessor.quick_map
    DatasetAccessor.roi
    DatasetAccessor.subset
    DatasetAccessor.transform
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
