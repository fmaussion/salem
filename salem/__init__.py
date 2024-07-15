"""Salem package."""

import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import pyproj

from .version import __version__


def lazy_property(fn: Callable) -> Callable:
    """Lazy-evaluate a property (Decorator)."""
    attr_name = '_lazy_' + fn.__name__

    @property
    @wraps(fn)
    def _lazy_property(self: object) -> object:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


# Default proj
wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')

# Path to the cache directory
cache_dir = Path.home() / '.salem_cache'
cache_dir.mkdir(exist_ok=True)
download_dir = cache_dir / 'downloads'
download_dir.mkdir(exist_ok=True)

sample_data_gh_commit = '454bf696324000d198f574a1bf5bc56e3e489051'
sample_data_dir = cache_dir / f'salem-sample-data-{sample_data_gh_commit}'

# python version
python_version = 'py3'
if sys.version_info.major == 2:
    python_version = 'py2'

# API
from salem.datasets import (
    WRF,
    EsriITMIX,
    GeoDataset,
    GeoNetcdf,
    GeoTiff,
    GoogleCenterMap,
    GoogleVisibleMap,
)
from salem.gis import (
    Grid,
    check_crs,
    googlestatic_mercator_grid,
    mercator_grid,
    proj_is_latlong,
    proj_is_same,
    proj_to_cartopy,
    transform_geometry,
    transform_geopandas,
    transform_proj,
)
from salem.sio import (
    DataArrayAccessor,
    DatasetAccessor,
    grid_from_dataset,
    open_metum_dataset,
    open_mf_wrf_dataset,
    open_wrf_dataset,
    open_xr_dataset,
    read_shapefile,
    read_shapefile_to_grid,
)
from salem.utils import get_demo_file, reduce

try:
    from salem.graphics import DataLevels, Map, get_cmap
except ImportError as err:
    if 'matplotlib' not in str(err):
        raise

    def get_cmap() -> None:
        msg = 'requires matplotlib'
        raise ImportError(msg)

    def DataLevels() -> None:
        msg = 'requires matplotlib'
        raise ImportError(msg)

    def Map() -> None:
        msg = 'requires matplotlib'
        raise ImportError(msg)


from salem.wrftools import geogrid_simulator
