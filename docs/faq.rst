.. _faq:

Frequently Asked Questions
==========================

.. _faqtools:

What others tools should I know about?
--------------------------------------

If you want to plot on maps, `cartopy`_ is arguably one of the best tool you
could pick. For real reprojection workflows on big gridded files you probably
want to use `rasterio`_.

The python atmopsheric sciences community is a bit spread between `iris`_ and
`xarray`_ for N-Dimensional data handling (I picked the later for it's
strong interaction with `pandas`_). Several great libraries are available to
meteorologists and climatologists, for example `MetPy`_,
`windspharm`_, `xgcm`_, and others will come.
Let me know if I forgot something!


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
GDAL, but since `conda-forge`_  GDAL is much easier to install.

Also, I use the atmospheric model WRF in my work, and its output files are
absolutely NOT compliant with the CF conventions. To my knowledge,
there is no tool to analyse WRF data with Python, and Salem will be
further developed with this model in mind.

.. _conda-forge: http://conda-forge.github.io/


Why aren't you using Cartopy for your maps?
-------------------------------------------

This is actually a good question, and I think that there are no obstacle for
Salem to use it's geolocation information to plot gridded data on cartopy's
maps. It's just that I never really got to understand how cartopy really works.
Want to do a PR?

Furthermore, I kind of like how Salem's maps look, and (since
I've coded it), I find it nice and easy to use. But that, of course, might not
be your opinion.


What's this "salem_cache" directory in my home folder?
------------------------------------------------------

At the first import, Salem will create a hidden directory called
``.salem_cache`` in your home folder. It will be used to download Salem's
demo files and standard shapefiles. This directory is also used by
joblib to store the result of slow operations such as reading and
transforming shapefiles, or downloading google maps from the internet. The
cache should not become too large, but if it does: simply delete it.