.. _faq:

Frequently Asked Questions
==========================

Is your library mature for production code?
-------------------------------------------

No. The API is not always as clever as I wish it would, and it will probably
change in the future. It is quite well tested though, at least for the cases
I encounter in my daily work.


.. _faqtools:

What others tools should I know about?
--------------------------------------

If you want to plot on maps, `cartopy`_ is probably one of the best tools you
could pick. For reprojection workflows on large or numerous gridded files you
probably want to use `rasterio`_.

The python atmopsheric sciences community is a bit spread between `iris`_ and
`xarray`_ for N-Dimensional data handling (I picked the later for it's
strong interaction with `pandas`_). Several great libraries are available to
meteorologists and climatologists, for example `MetPy`_,
`windspharm`_, `xgcm`_, and all the ones I forgot to mention.

.. _cartopy: http://scitools.org.uk/cartopy/docs/latest/index.html
.. _rasterio: https://github.com/mapbox/rasterio
.. _iris: http://scitools.org.uk/iris/
.. _xarray: http://xarray.pydata.org/en/stable/
.. _pandas: http://pandas.pydata.org/
.. _windspharm: http://ajdawson.github.io/windspharm/
.. _xgcm: https://github.com/xgcm/xgcm
.. _MetPy: http://metpy.readthedocs.io/en/stable/


But then, why developing Salem?
-------------------------------

As an atmospheric scientist, I hate to have to take care about projections and
maps. Salem was created to hide all these concerns. By the time I started, it
seemed a good idea to provide map transformation tools without depending on
GDAL (thanks to `conda-forge`_  GDAL is now much easier to install).
It is still possible to do reprojection work in Salem using scipy and
pyproj alone. It is al done in python and on memory,
so don't expect miracles on that side.

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
