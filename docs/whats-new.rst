.. currentmodule:: salem

What's New
==========

v0.3.X (Unreleased)
-------------------


v0.3.7 (08 November 2021)
-------------------------

A minor release of the salem package to fix deprecation warnings from shapely,
and remove the dependency to the discontinued package "descartes".

v0.3.6 (29 October 2021)
------------------------

A minor release of the salem package to fix a version number issue in v0.3.5
and a compatibility fix for cartopy > 0.20.

v0.3.5 (29 August 2021)
-----------------------

A minor release of the salem package to fix for updates in xarray and
an update in the Natural Earth image url.

v0.3.4 (24 March 2021)
----------------------

A minor release of the salem package to fix for updates in
xarray.

v0.3.3 (18 November 2020)
-------------------------

A minor release of the salem package to make up for bugs in 0.3.2 and 0.3.1
that happened because of tiredness.

v0.3.1 (18 November 2020)
-------------------------

A minor release of the salem package to make up for a version problem
in v0.3.0.


v0.3.0 (20 August 2020)
-----------------------

A minor release of the salem package, mostly for bug fixes and small
enhancements. **We dropped support for Python 2.**

Enhancements
~~~~~~~~~~~~

- Region Of Interest (ROI) now also accepts GeoDataFrame objects
  (:pull:`161`).
- Added support for lossy compression on WRF files (:pull:`167`).

Bug fixes
~~~~~~~~~

- More accurate web mercator proj for google static images (:pull:`178`).
- Various fixes for upstream changes, mostly in pyrproj and geopandas
  (:pull:`148`, :pull:`150`, :pull:`154`, :pull:`156`, :pull:`164`).
- Fixes for changes in matplotlib (:pull:`172`, :pull:`176`)


v0.2.4 (25 March 2019)
------------------------

A minor release of the salem package with a fix to support xarray v0.12.
**This will (probably) be the last release to support python 2**.


v0.2.3 (06 January 2019)
------------------------

A minor release of the salem package with small bugfixes and a couple of
enhancements. **This will be the last release to support python 2**.

Enhancements
~~~~~~~~~~~~

- new :py:func:`~Grid.extent_as_polygon` method, which creates a polygon
  outlining the contours of a Grid.
- minimal support for MetUm rotated grid files
  (:py:func:`~salem.open_metum_dataset`), still experimental (:pull:`117`).


Bug fixes
~~~~~~~~~

- the xarray accessor method ``roi()`` now preserves encoding (:issue:`109`).
  By `Johannes Landmann <https://github.com/jlandmann>`_
- fixed a bug in ``geogrid_simulator`` that would occur if `stand_lon` and
  `ref_lon` were not set to the same value (a strange idea tbh)


v0.2.2 (21 June 2018)
---------------------

A long overdue minor release of the salem package. It contains several
enhancements and bugfixes.

Enhancements
~~~~~~~~~~~~

- :py:func:`~transform_geopandas` can now handle grid to proj transformations.
- Thw WRF vertical interpolation methods now accept a `fill_value` kwarg
  which is `np.NaN` per default but can be set to `'extrapolate'` if
  needed.
- New :py:func:`~reduce` function, useful to aggregate structured high-res
  grids to lower-res grids (:ref:`sphx_glr_auto_examples_plot_subgrid_mask.py`)
- new :py:func:`~Grid.to_geometry` method, useful to compute precise
  vector to raster masks (TODO: example showing its use)
- new projection for WRF files: polar stereographic
- one can now add a scale bar to maps (see :py:func:`~Map.set_scale_bar`)
- google static maps now support the `scale` kwarg for higher resolution
  images (:pull:`91`).
  By `tbridel <https://github.com/tbridel>`_
- each salem version is now pinned to a certain commit of the sample-data
  repository. This is more robust and will avoid future tests to fail
  for the wrong reasons.
- accessor's ``roi`` method now accepts an ``other`` kwarg to fill masked
  values with something else than Nan (:pull:`96`).
  By `Schlump <https://github.com/Schlump>`_


Bug fixes
~~~~~~~~~

- the cache directory is also updated when the ``pandas`` version changes
  (:issue:`74`)
- small bugfixes in the projections and warning handling
- PRESSURE variable was given in Pa, not hPa. This is corrected now.
- Small bugfixes in exotic WRF diag variables indexing
- Corrected a bug in `proj_to_cartopy` where the spherical parameters were
  silently ignored. Cartopy maps on WRF domains are now perfect!
- ``Grid.ij_to_crs`` should now handle non numpy arrays too
- Corrected an internal bug with latest xarray version (:pull:`97`). Users
  will have to update to latest xarray version (v0.10+).
- Added new coordinate names to be understood by salem (:issue:`100`).


v0.2.1 (07 February 2017)
-------------------------

Minor release containing a couple of enhancements and deprecations, but mostly
fixes for bugs and for the recent xarray and matplotlib upgrades.

Deprecations
~~~~~~~~~~~~

- the ``corner``, ``ul_corner``, and ``ll_corner`` kwargs to grid
  initialisation have been deprecated in favor of the more explicit ``x0y0``.
- the Grid attribue ``order`` is now called ``origin``, and can have the values
  ``"lower-left"`` or ``"upper-left"``
- ``salem.open_xr_dataset`` now succeeds if and only if the grid is understood
  by salem. This makes more sense, as unexpected errors should not be
  silently ignored

Enhancements
~~~~~~~~~~~~

- new :py:func:`~open_mf_wrf_dataset` function
- new ``deacc`` method added to DataArrayAccessors
  By `SiMaria <https://github.com/SiMaria>`
- new ``Map.transform()`` method to make over-plotting easier (experimental)
- new **all_touched** argument added to ``Grid.region_of_interest()`` to pass
  through to  ``rasterio.features.rasterize``, tweaking how grid cells are
  included in the region of interest.
  By `Daniel Rothenberg <https://github.com/darothen>`_
- new :py:func:`~DataArrayAccessor.lookup_transform` projection transform
- ``Grid`` objects now have a decent ``__repr__``
- new serialization methods for grid: ``to_dict``, ``from_dict``, ``to_json``,
  ``from_json``.


Bug fixes
~~~~~~~~~

- ``grid.transform()`` now works with non-numpy array type
- ``transform_geopandas()`` won't do inplace operation per default anymore
- fixed bugs to prepare for upcoming xarray v0.9.0
- ``setup.py`` makes cleaner version strings for development versions (like
  xarray's)
- joblib's caching directory is now "environment dependant" in order  to avoid
  conda/not conda mismatches, and avoid other black-magic curses related to
  caching.
- fixed a bug with opening geo_em files using xarray or open_wrf_dataset
- various bug fixes and enhancements to the graphics (including the
  relocation of testing baseline images to the demo repo)


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
