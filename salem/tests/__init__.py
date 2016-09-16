from __future__ import division
import unittest

try:
    import xarray
    has_xarray = True
except ImportError:
    has_xarray = False

try:
    import shapely
    has_shapely = True
except ImportError:
    has_shapely = False


def requires_xarray(test):
    # Test decorator
    msg = "requires xarray"
    return test if has_xarray else unittest.skip(msg)(test)


def requires_shapely(test):
    # Test decorator
    msg = "requires shapely"
    return test if has_shapely else unittest.skip(msg)(test)