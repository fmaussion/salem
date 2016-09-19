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

    graphics.get_cmap
    graphics.DataLevels
    graphics.Map


Input/output
============

.. autosummary::
    :toctree: generated/
    :nosignatures:

    read_shapefile
    read_shapefile_to_grid
    XarrayAccessor
    open_xr_dataset
    utils.get_demo_file


Old-style datasets
==================

Old-style Datasets (prior to xarray), kept for backwards compatibility
reasons. Eventually, they will be replaced by xarray's datasets.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    GeoDataset
    GeoTiff
    GeoNetcdf
    WRF
    GoogleCenterMap
    GoogleVisibleMap
