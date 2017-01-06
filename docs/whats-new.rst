.. currentmodule:: salem

What's New
==========


v0.2.X (Unreleased)
-------------------

Enhancements
~~~~~~~~~~~~

- new ``open_mf_wrf_dataset`` function
- new ``deacc`` method added to DataArrayAccessors
- new ``Map.transform()`` method to make over-plotting easier (experimental)
- new **rasterize_kws** argument added to ``Grid.region_of_interest()`` to pass additional kwargs to  ``rasterio.features.rasterize``

Bug fixes
~~~~~~~~~

- ``grid.transform()`` now works with non numpy array type
- ``transform_geopandas()`` won't do inplace operation per default anymore


v0.2.0 (08 November 2016)
-------------------------

Salem is now released under a 3-clause BSD license.

Enhancements
~~~~~~~~~~~~

- New :py:func:`~DatasetAccessor.wrf_zlevel` and
  :py:func:`~DatasetAccessor.wrf_plevel` functions for vertical interpolation
- Salem can now plot on cartopy's maps thanks to a new
  :py:func:`~salem.proj_to_cartopy` function.
- Doc improvements
- New diagnostic variable: 'WS'


v0.1.1 (27 October 2016)
------------------------

Enhancements
~~~~~~~~~~~~

- Some doc improvements
- New ``ds`` keyword to the accessors ``subset()`` and ``roi()`` methods

Bug fixes
~~~~~~~~~

- Natural Earth file `lr` (low-res) now shipped with sample-data, `mr` and `hr`
  can still be downloaded if needed
- Remove use to deprecated ``rasterio.drivers()``
- GeoNetCDF files without time variable should now open without error


v0.1.0 (22 October 2016)
------------------------

Big refactoring (:pull:`15`), partly backwards incompatible (mostly renaming).
Improved xarray accessors, WRF tools, merged `Cleo`_ into the codebase,
added a first draft of documentation.

.. _Cleo: https://github.com/fmaussion/cleo


v0.0.9 (22 October 2016)
------------------------

Initial release.
