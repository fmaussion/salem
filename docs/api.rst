#############
API reference
#############

Here we will add the documentation for selected modules.

.. currentmodule:: salem

Georeferencing
==============

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Grid
    check_crs
    proj_is_same
    transform_proj
    transform_geometry
    transform_geopandas
    mercator_grid

Graphics
========

.. autosummary::
    :toctree: generated/
    :nosignatures:

    get_cmap
    DataLevels
    Map


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
