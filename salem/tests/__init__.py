from __future__ import division
import unittest
from distutils.version import LooseVersion
import os
from salem import python_version
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError

on_travis = False
if os.environ.get('TRAVIS') is not None:
    on_travis = True


def has_internet():
    """Not so recommended it seems"""
    try:
        _ = urlopen('http://www.google.com', timeout=1)
        return True
    except URLError:
        pass
    return False

try:
    import shapely
    has_shapely = True
except ImportError:
    has_shapely = False

try:
    import geopandas
    has_geopandas = True
except ImportError:
    has_geopandas = False

try:
    import motionless
    has_motionless = True
except ImportError:
    has_motionless = False

try:
    import matplotlib
    mpl_version = LooseVersion(matplotlib.__version__)
    has_matplotlib = mpl_version >= LooseVersion('2')
except ImportError:
    has_matplotlib = False
    mpl_version = LooseVersion('0.0.0')

try:
    import rasterio
    has_rasterio = True
except ImportError:
    has_rasterio = False

try:
    import cartopy
    has_cartopy = True
except ImportError:
    has_cartopy = False

try:
    import dask
    has_dask = True
except ImportError:
    has_dask = False


def requires_internet(test):
    msg = "requires internet"
    return test if has_internet() else unittest.skip(msg)(test)


def requires_matplotlib_and_py3(test):
    msg = "requires matplotlib and py3"
    return test if has_matplotlib and (python_version == 'py3') \
        else unittest.skip(msg)(test)


def requires_matplotlib(test):
    msg = "requires matplotlib"
    return test if has_matplotlib else unittest.skip(msg)(test)


def requires_motionless(test):
    msg = "requires motionless"
    return test if has_motionless else unittest.skip(msg)(test)


def requires_rasterio(test):
    msg = "requires rasterio"
    return test if has_rasterio else unittest.skip(msg)(test)


def requires_cartopy(test):
    msg = "requires cartopy"
    return test if has_cartopy else unittest.skip(msg)(test)


def requires_shapely(test):
    msg = "requires shapely"
    return test if has_shapely else unittest.skip(msg)(test)


def requires_geopandas(test):
    msg = "requires geopandas"
    return test if has_geopandas else unittest.skip(msg)(test)


def requires_dask(test):
    msg = "requires dask"
    return test if has_dask else unittest.skip(msg)(test)


def requires_travis(test):
    msg = "requires travis"
    return test if on_travis else unittest.skip(msg)(test)
