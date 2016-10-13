"""
Salem package

Copyright: Fabien Maussion, 2014-2016

License: GPLv3
"""
from __future__ import division

from os import path
from os import makedirs
import sys
from functools import wraps

import pyproj

try:
    from .version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('Salem is not properly installed. If you are running '
                      'from the source directory, please instead create a '
                      'new virtual environment (using conda or virtualenv) '
                      'and  then install it in-place by running: '
                      'pip install -e .')


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""

    attr_name = '_lazy_' + fn.__name__

    @property
    @wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property

# Default proj
wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')

# Path to the cache directory
cache_dir = path.join(path.expanduser('~'), '.salem_cache')
if not path.exists(cache_dir):
    makedirs(cache_dir)
download_dir = path.join(cache_dir, 'downloads')
if not path.exists(download_dir):
    makedirs(download_dir)

# python version
python_version = 'py3'
if sys.version_info.major == 2:
    python_version = 'py2'

# API
from salem.gis import *
from salem.datasets import *
from salem.sio import read_shapefile, read_shapefile_to_grid
from salem.sio import open_xr_dataset, open_wrf_dataset, \
    DataArrayAccessor, DatasetAccessor
from salem.utils import get_demo_file
from salem.graphics import get_cmap, DataLevels, Map
from salem.wrftools import geogrid_simulator

