.. _faq:

Frequently Asked Questions
==========================

What is the development status of the salem library?
----------------------------------------------------

As of today (August 2021), salem is used by several people (number unknown)
and is used by at least one downstream larger project
(`OGGM <https://oggm.org>`_). I plan to continue to maintain salem in the
future, but cannot spend much time and energy in new, larger features that
the community might need. These larger features (mostly: improved support
for more datasets and improved plotting) should be carried out by better and
more adopted projects (mostly: `geoxarray`_ and `cartopy`_).

Salem is small but well tested for the cases I encounter in my daily work.
I don't think that salem will become a major library (there are so many
great projects out there!), but I think it will be a useful complement for a
few. For more information on my motivations to develop salem,
see :ref:`whysalem` and
`this github discussion <https://github.com/geoxarray/geoxarray/issues/3>`_.

.. _geoxarray: https://github.com/geoxarray/geoxarray

How should I cite salem?
------------------------

If you are using salem and would like to cite it in academic publication, we
would certainly appreciate it. We recommend to use the zenodo DOI for
this purpose:

    .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596573.svg
       :target: https://doi.org/10.5281/zenodo.596573

An example BibTeX entry::

    @software{salem,
      author       = {Fabien Maussion and
                      TimoRoth and
                      Johannes Landmann and
                      Matthias Dusch and
                      Ray Bell and
                      tbridel},
      title        = {fmaussion/salem: v0.3.4},
      month        = mar,
      year         = 2021,
      publisher    = {Zenodo},
      version      = {v0.3.4},
      doi          = {10.5281/zenodo.4635291},
      url          = {https://doi.org/10.5281/zenodo.4635291}
    }


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
- `wrf-python`_ provides much more WRF functionalities than salem. It is the
  recommended package to do computations with WRF output. Salem's syntax is
  nicer than that of wrf-python, though.
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
.. _wrf-python: https://wrf-python.readthedocs.io
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

.. _whysalem:

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
`wrf-python`_ is a great tool to manipulate WRF data with Python, but it also
have several drawbacks (mostly: its syntax). Salem will be further developed
with the WRF model in mind.

.. _conda-forge: http://conda-forge.github.io/


Why is my dataset not supported? (`dataset Grid not understood` error)
----------------------------------------------------------------------

As salem gained in visibility, we started to get requests to support new
dataset formats (see the corresponding
`github issue <https://github.com/fmaussion/salem/issues/100>`_). While I am
generally in favor of supporting new datasets, it will be impossible to support
all of them in an automated manner. Here Ill try to explain why.

Salem works in the cartesian, map projection space. This means, it needs to
understand the data's map projection and the name of the eastings, northings
coordinates in that projection. Most datasets (especially from older models)
use their own (bad) naming convention for things, and these names and
conventions have to hard-coded in salem. To my knowledge there is no
automated parser of geospatial information in python: `geoxarray`_ is a
currently staled attempt to do so.

Salem doesn't make use of the 2D lon/lat coordinates on the globe (when it does
it's just for testing). Working in the projected space has several advantages,
mostly for performance and precision reasons.

Note that `some people don't agree with this view`_, and don't care about the
projection of their data as long as they have access to the 2D lon/lat
coordinates. xarray (with `cartopy`_) can plot data based on their 2D
coordinates, and `xesmf <https://xesmf.readthedocs.io/>`_ performs
regridding on the globe without worrying about map projections. These
tools are maybe the right tools for you!

.. _some people don't agree with this view: https://github.com/pangeo-data/pangeo/issues/356#issuecomment-415168433


What's this ".salem_cache" directory in my home folder?
-------------------------------------------------------

At the first import, Salem will create a hidden directory called
``.salem_cache`` in your home folder. It will be used to download Salem's
demo files and standard shapefiles. This directory is also used by
joblib to store the result of slow operations such as reading and
transforming shapefiles, or downloading google maps from the internet. The
cache should not become too large, but if it does: simply delete it.
