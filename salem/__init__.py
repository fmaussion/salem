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

import pyproj


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""

    attr_name = '_lazy_' + fn.__name__

    @property
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
