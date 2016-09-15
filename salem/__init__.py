"""
Salem
=====

Provides some GIS and data handling tools.

Copyright: Fabien Maussion, 2014-2015

License: GPLv3+
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

wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')

# Path to the cache directory
cache_dir = path.join(path.expanduser('~'), '.salem_cache')
if not path.exists(cache_dir):
    makedirs(cache_dir)  # pragma: no cover

# python version
python_version = 'py3'
if sys.version_info.major == 2:
    python_version = 'py2'  # pragma: no cover

# API
from salem.gis import *
from salem.datasets import *
