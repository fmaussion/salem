What's New
==========


v0.x (unreleased)
-----------------

Enhancements
~~~~~~~~~~~~

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
