.. _faq:

Frequently Asked Questions
==========================

Is your library mature for production code?
-------------------------------------------

Not really. The API is not always as clever as I wish it would, and it will
probably change in the future. Salem is well tested though, at least for the
cases I encounter in my daily work.


.. _faqtools:

What others tools should I know about?
--------------------------------------

The python atmospheric sciences community is a bit spread between `iris`_ and
`xarray`_ for N-Dimensional data handling. I find that xarray is very intuitive
to learn thanks to its strong interaction with `pandas`_.

Here are some tools that share functionalities with Salem:

- `cartopy`_ is the reference tool for plotting on maps. Salem provides a way
  to plot with cartopy in addition to Salem's homegrowm graphics.
  (see :ref:`plotting`)
- Salem provides useful reprojection tools (see :ref:`gis`). The transformation
  routines are quite fast (we use pyproj for the map transformations and
  scipy for the interpolation) but they are all done on memory (i.e. not
  adapted for large datasets). For large reprojection workflows you might want
  to have a look at `cartopy`_ and `pyresample`_.
- `regionmask`_ provides similar tools as salem's region-of-interest
  functionalities if you are woking with shapefiles. regionmask seems a bit
  more general than Salem, but I'm not sure if it works with any map
  projection as Salem does.
- In the future, I hope that `pangeo-data`_ will overtake most of the
  functionalities I need. But this is not going to happen tomorrow...


Several libraries are available to meteorologists and climatologists, but I
don't think they share much functionality with Salem: for example `MetPy`_,
`windspharm`_, `xgcm`_, `aospy`_, and all the ones I forgot to mention.

.. _cartopy: http://scitools.org.uk/cartopy/docs/latest/index.html
.. _pyresample: https://github.com/pytroll/pyresample
.. _rasterio: https://github.com/mapbox/rasterio
.. _iris: http://scitools.org.uk/iris/
.. _xarray: http://xarray.pydata.org/en/stable/
.. _pandas: http://pandas.pydata.org/
.. _windspharm: http://ajdawson.github.io/windspharm/
.. _xgcm: https://github.com/xgcm/xgcm
.. _MetPy: http://metpy.readthedocs.io/en/stable/
.. _aospy: https://github.com/spencerahill/aospy
.. _regionmask: https://github.com/mathause/regionmask
.. _pangeo-data: https://pangeo-data.github.io/


Why developing Salem?
---------------------

As an atmospheric scientist, I hate to have to take care about projections and
maps. Salem was created to hide all these concerns. By the time I started, it
seemed a good idea to provide map transformation tools without depending on
GDAL (thanks to `conda-forge`_  GDAL is now much easier to install).
It is still possible to do reprojection work in Salem using scipy and
pyproj alone.

Furthermore, I use the atmospheric model WRF quite often in my work.
Its output files are absolutely not compliant with the CF conventions.
To my knowledge, there is no tool to plot and manipulate WRF data with Python,
and Salem will be further developed with this model in mind.

.. _conda-forge: http://conda-forge.github.io/


What's this ".salem_cache" directory in my home folder?
-------------------------------------------------------

At the first import, Salem will create a hidden directory called
``.salem_cache`` in your home folder. It will be used to download Salem's
demo files and standard shapefiles. This directory is also used by
joblib to store the result of slow operations such as reading and
transforming shapefiles, or downloading google maps from the internet. The
cache should not become too large, but if it does: simply delete it.
