"""Input output functions (but mostly input)."""

from __future__ import annotations

import contextlib
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cftime
import netCDF4
import numpy as np
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4DataStore
from xarray.core import dtypes

from salem import gis, utils, wgs84, wrftools
from salem.gis import check_crs, proj_to_cartopy
from salem.utils import (
    cached_shapefile_path,
    deprecated_arg,
    import_if_exists,
    memory,
)

if TYPE_CHECKING:
    import threading

    import pandas as pd
    from matplotlib import axes
    from numpy._typing import NDArray

    from salem.graphics import Map

try:
    from xarray.backends.locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks

    netcdf4_python_lock = combine_locks([NETCDFC_LOCK, HDF5_LOCK])
except ImportError:
    # xarray < v0.11
    from xarray.backends.api import _default_lock as netcdf4_python_lock
try:
    from xarray.core.pycompat import basestring
except ImportError:
    # latest xarray dropped python2 support, so we can safely assume py3 here
    basestring = str

# Locals
from salem.gis import transform_proj

has_cartopy = import_if_exists('cartopy')
if has_cartopy:
    from cartopy import crs
has_geopandas = import_if_exists('geopandas')
if has_geopandas:
    from geopandas import GeoDataFrame, read_file

tolerance_by_proj_id = {
    1: 1e-3,
    2: 5e-3,
    3: 1e-3,
    6: 1e-3,
}


def read_shapefile(fpath: Path, *, cached: bool = False) -> GeoDataFrame:
    """Read a shapefile using geopandas.

    For convenience, it adds four columns to the dataframe:
    [min_x, max_x, min_y, max_y]

    Because reading a shapefile can take a long time, Salem provides a
    caching utility (cached=True). This will save a pickle of the shapefile
    in the cache directory.
    """
    if not has_geopandas:
        msg = 'read_shapefile requires geopandas to be installed'
        raise RuntimeError(msg)
    if fpath.suffix.lower() in ['.shp', '.p']:
        if cached:
            cpath = cached_shapefile_path(fpath)
            # unpickle if cached, read and pickle if not
            if cpath.exists():
                with cpath.open('rb') as f:
                    return pickle.load(f)
            out = read_shapefile(fpath, cached=False)
            with cpath.open('wb') as f:
                pickle.dump(out, f)
            return out
        out = read_file(fpath)
        out['min_x'] = [g.bounds[0] for g in out.geometry]
        out['max_x'] = [g.bounds[2] for g in out.geometry]
        out['min_y'] = [g.bounds[1] for g in out.geometry]
        out['max_y'] = [g.bounds[3] for g in out.geometry]
        return out
    msg = f'File extension not recognised: {fpath.suffix}'
    raise ValueError(msg)


@memory.cache(ignore=['grid'])
def _memory_shapefile_to_grid(
    shape_cpath: Path, grid: gis.Grid, **kwargs
) -> gis.Grid:
    """Quick solution to not transform many times the same shape (useful for maps).

    Using joblib. Since grid is a complex object, joblib seems to have trouble with it.
    So joblib is checking its cache according to the grid params while the job
    is done with grid.
    """
    shape = read_shapefile(shape_cpath, cached=True)
    e = grid.extent_in_crs(crs=shape.crs)
    p = np.nonzero(
        ~(
            (shape['min_x'].to_numpy() > e[1])
            | (shape['max_x'].to_numpy() < e[0])
            | (shape['min_y'].to_numpy() > e[3])
            | (shape['max_y'].to_numpy() < e[2])
        )
    )
    shape = shape.iloc[p]
    return gis.transform_geopandas(shape, to_crs=grid, inplace=True)


def read_shapefile_to_grid(fpath: Path, grid: gis.Grid) -> gis.Grid:
    """Read a shapefile and transform to salem.Grid object.

    Same as read_shapefile but directly transformed to a grid.
    The whole thing is cached so that the second call will
    will be much faster.

    Parameters
    ----------
    fpath: path to the file
    grid: the arrival grid

    Returns
    -------
    a salem.Grid object

    """
    # ensure it is a cached pickle (copy code smell)
    shape_cpath = cached_shapefile_path(fpath)
    if not shape_cpath.exists():
        out = read_shapefile(fpath, cached=False)
        with shape_cpath.open('wb') as f:
            pickle.dump(out, f)

    return _memory_shapefile_to_grid(shape_cpath, grid=grid, **grid.to_dict())


# TODO: remove this once we sure that we have all WRF files right
tmp_check_wrf = True


def _wrf_grid_from_dataset(ds: xr.Dataset) -> gis.Grid:
    """Get the WRF projection out of the file."""
    pargs = {}
    if hasattr(ds, 'PROJ_ENVI_STRING'):
        # HAR and other TU Berlin files
        dx = ds.GRID_DX if hasattr(ds, 'GRID_DX') else ds.DX
        dy = ds.GRID_DY if hasattr(ds, 'GRID_DY') else ds.DY
        if ds.PROJ_NAME in [
            'Lambert Conformal Conic',
            'WRF Lambert Conformal',
        ]:
            proj_id = 1
            pargs['lat_1'] = ds.PROJ_STANDARD_PAR1
            pargs['lat_2'] = ds.PROJ_STANDARD_PAR2
            pargs['lat_0'] = ds.PROJ_CENTRAL_LAT
            pargs['lon_0'] = ds.PROJ_CENTRAL_LON
            pargs['center_lon'] = ds.PROJ_CENTRAL_LON
        elif ds.PROJ_NAME in ['lat-lon']:
            proj_id = 6
        elif 'mercator' in ds.PROJ_NAME.lower():
            proj_id = 3
            pargs['lat_ts'] = ds.TRUELAT1
            pargs['center_lon'] = ds.CEN_LON
        else:
            proj_id = 99  # pragma: no cover
    else:
        # Normal WRF file
        cen_lon = ds.CEN_LON
        cen_lat = ds.CEN_LAT
        dx = ds.DX
        dy = ds.DY
        pargs['lat_1'] = ds.TRUELAT1
        pargs['lat_2'] = ds.TRUELAT2
        pargs['lat_0'] = ds.MOAD_CEN_LAT
        pargs['lon_0'] = ds.STAND_LON
        pargs['center_lon'] = ds.CEN_LON
        proj_id = ds.MAP_PROJ

    wrf_projection_key = {
        'Lambert Conformal': 1,
        'Polar Stereographic': 2,
        'Mercator': 3,
        'Lat-long': 6,
    }
    if proj_id == wrf_projection_key['Lambert Conformal']:
        # Lambert
        p4 = (
            f"+proj=lcc +lat_1={pargs['lat_1']} +lat_2={pargs['lat_2']} "
            f"+lat_0={pargs['lat_0']} +lon_0={pargs['lon_0']} "
            "+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )
    elif proj_id == wrf_projection_key['Polar Stereographic']:
        # Polar stereo
        p4 = (
            f"+proj=stere +lat_ts={pargs['lat_1']} +lon_0={pargs['lon_0']} +lat_0=90.0 "
            f"+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )
    elif proj_id == wrf_projection_key['Mercator']:
        # Mercator
        p4 = (
            f"+proj=merc +lat_ts={pargs['lat_1']} "
            f"+lon_0={pargs['center_lon']} "
            "+x_0=0 +y_0=0 +a=6370000 +b=6370000"
        )
    elif proj_id == wrf_projection_key['Lat-long']:
        # Lat-long
        p4 = f'+proj=eqc +lon_0={pargs['lon_0']} +x_0=0 +y_0=0 +a=6370000 +b=6370000'
    else:
        msg = f'WRF proj not understood: {proj_id}'
        raise NotImplementedError(msg)

    proj = gis.check_crs(p4)

    # Here we have to accept xarray and netCDF4 datasets
    try:
        nx = len(ds.dimensions['west_east'])
        ny = len(ds.dimensions['south_north'])
    except AttributeError:
        # maybe an xarray dataset
        nx = ds.sizes['west_east']
        ny = ds.sizes['south_north']
    if hasattr(ds, 'PROJ_ENVI_STRING'):
        # HAR
        x0 = float(ds['west_east'][0])
        y0 = float(ds['south_north'][0])
    else:
        # Normal WRF file
        e, n = gis.transform_proj(wgs84, proj, cen_lon, cen_lat)
        x0 = -(nx - 1) / 2.0 * dx + e  # DL corner
        y0 = -(ny - 1) / 2.0 * dy + n  # DL corner
    grid = gis.Grid(nxny=(nx, ny), x0y0=(x0, y0), dxdy=(dx, dy), proj=proj)

    if tmp_check_wrf:
        #  Temporary asserts
        if 'XLONG' in ds.variables:
            # Normal WRF
            reflon = ds.variables['XLONG']
            reflat = ds.variables['XLAT']
        elif 'XLONG_M' in ds.variables:
            # geo_em
            reflon = ds.variables['XLONG_M']
            reflat = ds.variables['XLAT_M']
        elif 'lon' in ds.variables:
            # HAR
            reflon = ds.variables['lon']
            reflat = ds.variables['lat']
        else:
            msg = "couldn't test for correct WRF lon-lat"
            raise RuntimeError(msg)

        if len(reflon.shape) == 3:
            reflon = reflon[0, :, :]
            reflat = reflat[0, :, :]
        mylon, mylat = grid.ll_coordinates

        atol = tolerance_by_proj_id[proj_id]
        check = np.isclose(reflon, mylon, atol=atol)
        if not np.all(check):
            n_pix = np.sum(~check)
            maxe = np.max(np.abs(reflon - mylon))
            if maxe < (360 - atol):
                msg = """
                    For {n_pix} grid points, the expected accuracy ({atol}) of our lons
                    did not match those of the WRF file. Max error: {maxe}
                    )""".format(n_pix=n_pix, atol=atol, maxe=maxe)
                deprecated_arg(msg)
        check = np.isclose(reflat, mylat, atol=atol)
        if not np.all(check):
            n_pix = np.sum(~check)
            maxe = np.max(np.abs(reflat - mylat))
            msg = """
                For {n_pix} grid points, the expected accuracy ({atol}) of our lats
                did not match those of the WRF file. Max error: {maxe}
                )""".format(n_pix=n_pix, atol=atol, maxe=maxe)
            deprecated_arg(msg)

    return grid


def _lonlat_grid_from_dataset(ds: xr.Dataset) -> gis.Grid | None:
    """Seek for longitude and latitude coordinates."""
    # Do we have some standard names as variable?
    keys_view = ds.variables.keys()
    vns = [str(k) for k in keys_view]
    xc = utils.str_in_list(vns, utils.valid_names['x_dim'])
    yc = utils.str_in_list(vns, utils.valid_names['y_dim'])

    # Sometimes there are more than one coordinates, one of which might have
    # more dims (e.g. lons in WRF files): take the first one with ndim = 1:
    x = None
    for xp in xc:
        if len(ds.variables[xp].shape) == 1:
            x = xp
    y = None
    for yp in yc:
        if len(ds.variables[yp].shape) == 1:
            y = yp
    if (x is None) or (y is None):
        return None

    # OK, get it
    lon = ds.variables[x][:]
    lat = ds.variables[y][:]

    # double check for dubious variables
    # name not usual. see if at least the range follows some conv
    if (
        not utils.str_in_list([x], utils.valid_names['lon_var'])
        or not utils.str_in_list([y], utils.valid_names['lat_var'])
    ) and ((np.max(np.abs(lon)) > 360.1) or (np.max(np.abs(lat)) > 90.1)):
        return None

    # Make the grid
    dx = lon[1] - lon[0]
    dy = lat[1] - lat[0]
    args = {
        'nxny': (lon.shape[0], lat.shape[0]),
        'proj': wgs84,
        'dxdy': (dx, dy),
        'x0y0': (lon[0], lat[0]),
    }
    return gis.Grid(**args)


def _salem_grid_from_dataset(ds: xr.Dataset) -> gis.Grid | None:
    """Seek for coordinates that Salem might have created.

    Current convention: x_coord, y_coord, pyproj_srs as attribute
    """
    # Projection
    try:
        proj = str(ds.pyproj_srs)
    except AttributeError:
        proj = None
    if proj is None:
        return None
    proj = gis.check_crs(proj)

    # Do we have some standard names as variable?
    keys_view = ds.variables.keys()
    vns = [str(k) for k in keys_view]
    xc = utils.str_in_list(vns, utils.valid_names['x_dim'])
    yc = utils.str_in_list(vns, utils.valid_names['y_dim'])

    # Sometimes there are more than one coordinates, one of which might have
    # more dims (e.g. lons in WRF files): take the first one with ndim = 1:
    x = None
    for xp in xc:
        if len(ds.variables[xp].shape) == 1:
            x = xp
    y = None
    for yp in yc:
        if len(ds.variables[yp].shape) == 1:
            y = yp
    if (x is None) or (y is None):
        return None

    # OK, get it
    x = ds.variables[x][:]
    y = ds.variables[y][:]

    # Make the grid
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    args = {
        'nxny': (x.shape[0], y.shape[0]),
        'proj': proj,
        'dxdy': (dx, dy),
        'x0y0': (x[0], y[0]),
    }
    return gis.Grid(**args)


def grid_from_dataset(ds: xr.Dataset) -> gis.Grid | None:
    """Find out if the dataset contains enough info for Salem to understand.

    ``ds`` can be an xarray dataset or a NetCDF dataset, or anything
    that resembles it.

    Returns a :py:class:`~salem.Grid` if successful, ``None`` otherwise
    """
    # try if it is a salem file
    out = _salem_grid_from_dataset(ds)
    if out is not None:
        return out

    # maybe it's a WRF file?
    if hasattr(ds, 'MOAD_CEN_LAT') or hasattr(ds, 'PROJ_ENVI_STRING'):
        # WRF and HAR have some special attributes
        return _wrf_grid_from_dataset(ds)

    # Try out platte carree
    return _lonlat_grid_from_dataset(ds)


def netcdf_time(
    ncobj: netCDF4.Dataset, *, monthbegin: bool = False
) -> pd.DatetimeIndex | None:
    """Check if the netcdf file contains a time that Salem understands."""
    import pandas as pd

    time = None
    try:
        keys_view = ncobj.variables.keys()
        vns = [str(k) for k in keys_view]
        vt = utils.str_in_list(vns, utils.valid_names['time_var'])[0]
    except IndexError:
        # no time variable
        return None

    if hasattr(ncobj, 'TITLE') and 'GEOGRID' in ncobj.TITLE:
        # geogrid file
        pass
    elif ncobj[vt].dtype in ['|S1', '|S19']:
        # WRF file
        time = []
        try:
            stimes = ncobj.variables['Times'][:].values
        except AttributeError:
            stimes = ncobj.variables['Times'][:]
        for t in stimes:
            time.append(
                pd.to_datetime(
                    t.tobytes().decode(),
                    errors='raise',
                    format='%Y-%m-%d_%H:%M:%S',
                )
            )
    elif vt is not None:
        # CF time
        var = ncobj.variables[vt]
        try:
            # We want python times because pandas doesn't understand
            # CFtime
            time = cftime.num2date(
                var[:],
                var.units,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
        except TypeError:
            # Old versions of cftime did return python times when possible
            time = cftime.num2date(var[:], var.units)

        if monthbegin:
            # sometimes monthly data is centered in the month (stupid)
            time = [
                datetime(t.year, t.month, 1, tzinfo=t.tzinfo) for t in time
            ]

    if time is None:
        return None
    return pd.DatetimeIndex(time)


class _XarrayAccessorBase:
    """Common logic for for both data structures (DataArray and Dataset).

    http://xarray.pydata.org/en/stable/internals.html#extending-xarray
    """

    def __init__(self, xarray_obj: xr.Dataset | xr.DataArray) -> None:
        self._obj = xarray_obj

        if isinstance(xarray_obj, xr.DataArray):
            xarray_obj = xarray_obj.to_dataset(name='var')
            # maybe there was already some georef
            with contextlib.suppress(Exception):
                xarray_obj.attrs['pyproj_srs'] = xarray_obj['var'].pyproj_srs

        self.grid = grid_from_dataset(xarray_obj)
        if self.grid is None:
            msg = 'dataset Grid not understood.'
            raise RuntimeError(msg)

        keys_view = xarray_obj.sizes.keys()
        dn = [str(k) for k in keys_view]
        self.x_dim = utils.str_in_list(dn, utils.valid_names['x_dim'])[0]
        self.y_dim = utils.str_in_list(dn, utils.valid_names['y_dim'])[0]
        dim = utils.str_in_list(dn, utils.valid_names['t_dim'])
        self.t_dim = dim[0] if dim else None
        dim = utils.str_in_list(dn, utils.valid_names['z_dim'])
        self.z_dim = dim[0] if dim else None

    def subset(
        self,
        margin: int = 0,
        ds: xr.Dataset | xr.DataArray | None = None,
        **kwargs,
    ) -> xr.Dataset | xr.DataArray:
        """Get a subset of the dataset.

        subset(self, margin=0, shape=None, geometry=None, grid=None,
                  corners=None, crs=wgs84, roi=None)

        Accepts all keywords of :py:func:`~Grid.roi`

        Parameters
        ----------
        margin : int
            add a margin to the region to subset (can be negative!). Can
            be used a single keyword, too: set_subset(margin=-5) will remove
            five pixels from each boundary of the dataset.
        ds : Dataset or DataArray
            form the ROI from the extent of the Dataset or DataArray
        shape : str
            path to a shapefile
        geometry : geometry
            a shapely geometry (don't forget the crs keyword)
        grid : Grid
            a Grid object which extent will form the ROI
        corners : tuple
            a ((x0, y0), (x1, y1)) tuple of the corners of the square
            to subset the dataset to  (don't forget the crs keyword)
        crs : crs, default wgs84
            coordinate reference system of the geometry and corners
        roi : ndarray
            a mask for the region of interest to subset the dataset onto

        """
        if ds is not None:
            grid = ds.salem.grid
            kwargs.setdefault('grid', grid)

        mask = self.grid.region_of_interest(**kwargs)
        if not kwargs:
            # user just wants a margin
            mask[:] = 1

        ids = np.nonzero(mask)
        sub_x = [np.min(ids[1]) - margin, np.max(ids[1]) + margin]
        sub_y = [np.min(ids[0]) - margin, np.max(ids[0]) + margin]

        return self._obj[
            {
                self.x_dim: slice(sub_x[0], sub_x[1] + 1),
                self.y_dim: slice(sub_y[0], sub_y[1] + 1),
            }
        ]

    def roi(
        self,
        ds: xr.Dataset | xr.DataArray | None = None,
        grid: gis.Grid | None = None,
        other: float | np.ndarray | xr.Dataset | xr.DataArray | None = None,
        **kwargs,
    ) -> xr.Dataset | xr.DataArray:
        """Make a region of interest (ROI) for the dataset.

        roi(self, shape=None, geometry=None, grid=None, corners=None,
               crs=wgs84, roi=None, all_touched=False, other=None)

        All grid points outside the ROI will be masked out.

        Parameters
        ----------
        ds : Dataset or DataArray
            form the ROI from the extent of the Dataset or DataArray
        shape : str
            path to a shapefile
        geometry : geometry
            a shapely geometry (don't forget the crs keyword)
        grid : Grid
            a Grid object which extent will form the ROI
        corners : tuple
            a ((x0, y0), (x1, y1)) tuple of the corners of the square
            to subset the dataset to  (don't forget the crs keyword)
        crs : crs, default wgs84
            coordinate reference system of the geometry and corners
        roi : ndarray
            if you have a mask ready, you can give it here
        all_touched : boolean
            pass-through argument for rasterio.features.rasterize, indicating
            that all grid cells which are  clipped by the shapefile defining
            the region of interest should be included (default=False)
        other : scalar, DataArray or Dataset, optional
            Value to use for locations in this object where cond is False. By
            default, these locations filled with NA.
            As in http://xarray.pydata.org/en/stable/generated/xarray.DataArray.where.html

        """
        other = kwargs.pop('other', dtypes.NA)
        if ds is not None:
            grid = ds.salem.grid
            kwargs.setdefault('grid', grid)

        mask = self.grid.region_of_interest(**kwargs)
        out = self._obj.where(mask, other=other)

        # keep attrs and encoding
        out.attrs = self._obj.attrs
        out.encoding = self._obj.encoding
        if isinstance(out, xr.Dataset):
            for v in self._obj.variables:
                out[v].encoding = self._obj[v].encoding
        if isinstance(out, xr.DataArray):
            out.name = self._obj.name

        # add pyproj string everywhere
        out.attrs['pyproj_srs'] = self.grid.proj.srs
        if isinstance(out, xr.Dataset):
            for v in out.data_vars:
                out.variables[v].attrs = self._obj.variables[v].attrs
                out.variables[v].attrs['pyproj_srs'] = self.grid.proj.srs
        return out

    def get_map(self, **kwargs) -> Map:
        """Make a salem.Map out of the dataset.

        All keywords are passed to :py:class:salem.Map
        """
        from salem.graphics import Map

        return Map(self.grid, **kwargs)

    def _quick_map(
        self,
        obj: xr.DataArray,
        ax: axes.Axes | None = None,
        interp: str = 'nearest',
        **kwargs,
    ) -> Map:
        """Make a plot of a data array."""
        # some metadata?
        title = str(obj.name) or ''
        title_add = obj._title_for_slice()
        if title_add:
            title += ' ({})'.format(str(title_add))
        cb = obj.attrs.get('units', '')

        smap = self.get_map(**kwargs)
        smap.set_data(obj.values, interp=interp)
        smap.visualize(ax=ax, title=title, cbar_title=cb)
        return smap

    def cartopy(self) -> crs.Projection:
        """Get a cartopy.crs.Projection for this dataset."""
        return proj_to_cartopy(self.grid.proj)

    def _apply_transform(
        self,
        transform: Callable,
        grid: gis.Grid,
        other: xr.Dataset | xr.DataArray,
        *,
        return_lut: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        """Apply common transform mixin."""
        was_dataarray = False
        if not isinstance(other, xr.Dataset):
            try:
                other = other.to_dataset(name=other.name)
                was_dataarray = True
            except AttributeError as att_err:
                # must be a ndarray
                if return_lut:
                    rdata, lut = transform(other, grid=grid, return_lut=True)
                else:
                    rdata = transform(other, grid=grid)
                    lut = None
                # let's guess
                sh = rdata.shape
                nd = len(sh)
                if nd == 2:
                    dims = (self.y_dim, self.x_dim)
                elif nd == 3:
                    newdim = 'new_dim'
                    if self.t_dim and sh[0] == self._obj.sizes[self.t_dim]:
                        newdim = self.t_dim
                    if self.z_dim and sh[0] == self._obj.sizes[self.z_dim]:
                        newdim = self.z_dim
                    dims = (newdim, self.y_dim, self.x_dim)
                else:
                    msg = 'more than 3 dims not ok yet.'
                    raise NotImplementedError(msg) from att_err

                coords = {}
                for d in dims:
                    if d in self._obj:
                        coords[d] = self._obj[d]

                out = xr.DataArray(rdata, coords=coords, dims=dims)
                out.attrs['pyproj_srs'] = self.grid.proj.srs
                if return_lut:
                    return out, lut
                return out

        # go
        out = xr.Dataset()
        for v in other.data_vars:
            var = other[v]
            if return_lut:
                rdata, lut = transform(var, return_lut=True)
            else:
                rdata = transform(var)

            # remove old coords
            dims = [d for d in var.dims]
            coords = {}
            for c in var.coords:
                n = utils.str_in_list([c], utils.valid_names['x_dim'])
                if n:
                    dims = [self.x_dim if x in n else x for x in dims]
                    continue
                n = utils.str_in_list([c], utils.valid_names['y_dim'])
                if n:
                    dims = [self.y_dim if x in n else x for x in dims]
                    continue
                coords[c] = var.coords[c]
            # add new ones
            coords[self.x_dim] = self._obj[self.x_dim]
            coords[self.y_dim] = self._obj[self.y_dim]

            rdata = xr.DataArray(
                rdata, coords=coords, attrs=var.attrs, dims=dims
            )
            rdata.attrs['pyproj_srs'] = self.grid.proj.srs
            out[v] = rdata

        if was_dataarray:
            out = out[v]
        else:
            out.attrs['pyproj_srs'] = self.grid.proj.srs

        if return_lut:
            return out, lut
        return out

    def transform(
        self,
        other: xr.Dataset | xr.DataArray | np.ndarray,
        grid: gis.Grid | None = None,
        interp: str = 'nearest',
        ks: int = 3,
    ) -> xr.Dataset | xr.DataArray:
        """Reprojects an other Dataset or DataArray onto self.

        The returned object has the same data structure as ``other`` (i.e.
        variables names, attributes), but is defined on the new grid
        (``self.grid``).

        Parameters
        ----------
        other: Dataset, DataArray or ndarray
            the data to project onto self
        grid: salem.Grid
            in case the input dataset does not carry georef info
        interp : str
            'nearest' (default), 'linear', or 'spline'
        ks : int
            Degree of the bivariate spline. Default is 3.

        Returns
        -------
        a dataset or a dataarray

        """
        transform = partial(self.grid.map_gridded_data, interp=interp, ks=ks)
        return self._apply_transform(transform, grid, other)

    def lookup_transform(
        self,
        other: xr.Dataset | xr.DataArray | np.ndarray,
        grid: gis.Grid | None = None,
        method: Callable = np.mean,
        lut: NDArray[Any] | None = None,
        *,
        return_lut: bool = False,
    ) -> xr.Dataset | xr.DataArray:
        """Project another Dataset or DataArray onto self via forward transform lookup.

        See : :py:meth:`Grid.lookup_transform`

        Parameters
        ----------
        other: Dataset, DataArray or ndarray
            the data to project onto self
        grid: salem.Grid
            in case the input dataset does not carry georef info
        method : function, default: np.mean
            the aggregation method. Possibilities: np.std, np.median, np.sum,
            and more. Use ``len`` to count the number of grid points!
        lut : ndarray, optional
            computing the lookup table can be expensive. If you have several
            operations to do with the same grid, set ``lut`` to an existing
            table obtained from a previous call to  :py:meth:`Grid.grid_lookup`
        return_lut : bool, optional
            set to True if you want to return the lookup table for later use.
            in this case, returns a tuple

        Returns
        -------
        a dataset or a dataarray
        If ``return_lut==True``, also return the lookup table

        """
        if grid is None:
            grid = check_crs(self.grid)
        transform = partial(self.grid.lookup_transform, method=method, lut=lut)
        return self._apply_transform(
            transform, grid, other, return_lut=return_lut
        )


@xr.register_dataarray_accessor('salem')
class DataArrayAccessor(_XarrayAccessorBase):
    """Salems xarray accessor for DataArrays."""

    def quick_map(
        self, ax: axes.Axes | None = None, interp: str = 'nearest', **kwargs
    ) -> Map:
        """Make a plot of the DataArray."""
        return self._quick_map(self._obj, ax=ax, interp=interp, **kwargs)

    def deacc(self, *, as_rate: bool = True) -> xr.DataArray:
        """De-accumulates the variable (i.e. compute the variable's rate).

        The returned variable has one element less over the time dimension.

        The default is to return in units per hour.

        Parameters
        ----------
        as_rate: bool
          set to false if you don't want units per hour,
          but units per given data timestep

        """
        out = self._obj[{self.t_dim: slice(1, len(self._obj[self.t_dim]))}]
        diff = self._obj[
            {self.t_dim: slice(0, len(self._obj[self.t_dim]) - 1)}
        ]
        out.values = out.to_numpy() - diff.to_numpy()
        out.attrs['description'] = out.attrs['description'].replace(
            'ACCUMULATED ', ''
        )

        if as_rate:
            dth = self._obj.time[1].to_numpy() - self._obj.time[0].to_numpy()
            dth = dth.astype('timedelta64[h]').astype(float)
            out.values = out.to_numpy() / dth
            out.attrs['units'] += ' h-1'
        else:
            out.attrs['units'] += ' step-1'

        return out

    def interpz(
        self,
        zcoord: xr.DataArray,
        levels: list[float],
        dim_name: str = '',
        fill_value: float = np.nan,
        *,
        use_multiprocessing: bool = True,
    ) -> xr.DataArray:
        """Interpolate the array along the vertical dimension.

        Parameters
        ----------
        zcoord: DataArray
          the z coordinates of the variable. Must be of same dimensions
        levels: 1dArray
          the levels at which to interpolate
        dim_name: str
          the name of the new dimension
        fill_value : np.nan or 'extrapolate', optional
          how to handle levels below the topography. Default is to mark them
          as invalid, but you might want the have them extrapolated.
        use_multiprocessing: bool
          set to false if, for some reason, you don't want to use mp

        Returns
        -------
        a new DataArray with the interpolated data

        """
        if self.z_dim is None:
            msg = 'zdimension not recognized'
            raise RuntimeError(msg)

        data = wrftools.interp3d(
            self._obj.values,
            zcoord.values,
            np.atleast_1d(levels),
            fill_value=fill_value,
            use_multiprocessing=use_multiprocessing,
        )

        dims = list(self._obj.dims)
        zd = np.nonzero([self.z_dim == d for d in dims])[0][0]
        dims[zd] = dim_name
        coords = dict(self._obj.coords)
        coords.pop(self.z_dim, None)
        coords[dim_name] = np.atleast_1d(levels)
        out = xr.DataArray(data, name=self._obj.name, dims=dims, coords=coords)
        out.attrs['pyproj_srs'] = self.grid.proj.srs
        if not np.asarray(levels).shape:
            out = out.isel(**{dim_name: 0})
        return out


@xr.register_dataset_accessor('salem')
class DatasetAccessor(_XarrayAccessorBase):
    """Salems xarray accessor for Datasets."""

    def quick_map(
        self,
        varname: str,
        ax: axes.Axes | None = None,
        interp: str = 'nearest',
        **kwargs,
    ) -> Map:
        """Make a plot of a variable of the DataSet."""
        return self._quick_map(
            self._obj[varname], ax=ax, interp=interp, **kwargs
        )

    def transform_and_add(
        self,
        other: xr.Dataset | xr.DataArray | np.ndarray,
        name: str | dict[str, str] | None = None,
        grid: gis.Grid | None = None,
        interp: str = 'nearest',
        ks: int = 3,
    ) -> xr.Dataset | xr.DataArray:
        """Reprojects an other Dataset and adds it's content to the current one.

        Parameters
        ----------
        other: Dataset, DataArray or ndarray
            the data to project onto self
        grid: salem.Grid
            in case the input dataset does not carry georef info
        interp : str
            'nearest' (default), 'linear', or 'spline'
        ks : int
            Degree of the bivariate spline. Default is 3.
        name: str or dict-like
            how to name to new variables in self. Per default the new variables
            are going to keep their name (it will raise an error in case of
            conflict). Set to a str to to rename the variable (if unique) or
            set to a dict for mapping the old names to the new names for
            datasets.

        """
        out = self.transform(other, grid=grid, interp=interp, ks=ks)

        if isinstance(out, xr.DataArray):
            new_name = name or out.name
            if new_name is None:
                msg = 'You need to set name'
                raise ValueError(msg)
            self._obj[new_name] = out
        elif isinstance(out, xr.Dataset):
            for v in out.data_vars:
                try:
                    new_name = name[v]
                except (KeyError, TypeError):
                    new_name = v
                self._obj[new_name] = out[v]
        return self._obj

    def wrf_zlevel(
        self,
        varname: str,
        levels: list[float] | NDArray[Any] | None = None,
        fill_value: float = np.nan,
        *,
        use_multiprocessing: bool = True,
    ) -> xr.Dataset:
        """Interpolates to a specified height above sea level.

        Parameters
        ----------
        varname: str
          the name of the variable to interpolate
        levels: 1d array
          levels at which to interpolate (default: some levels I thought of)
        fill_value : np.nan or 'extrapolate', optional
          how to handle levels below the topography. Default is to mark them
          as invalid, but you might want the have them extrapolated.
        use_multiprocessing: bool
          set to false if, for some reason, you don't want to use mp

        Returns
        -------
        an interpolated DataArray

        """
        if levels is None:
            levels = np.array(
                [
                    10,
                    20,
                    30,
                    50,
                    75,
                    100,
                    200,
                    300,
                    500,
                    750,
                    1000,
                    2000,
                    3000,
                    5000,
                    7500,
                    10000,
                ]
            )

        zcoord = self._obj['Z']
        out = self._obj[varname].salem.interpz(
            zcoord,
            levels,
            dim_name='z',
            fill_value=fill_value,
            use_multiprocessing=use_multiprocessing,
        )
        out['z'].attrs['description'] = 'height above sea level'
        out['z'].attrs['units'] = 'm'
        return out

    def wrf_plevel(
        self,
        varname: str,
        levels: list[float] | NDArray[Any] | None = None,
        fill_value: float = np.nan,
        *,
        use_multiprocessing: bool = True,
    ) -> xr.Dataset:
        """Interpolates to a specified pressure level (hPa).

        Parameters
        ----------
        varname: str
          the name of the variable to interpolate
        levels: 1d array, optional
          levels at which to interpolate (default: some levels I thought of)
        fill_value : np.nan or 'extrapolate', optional
          how to handle levels below the topography. Default is to mark them
          as invalid, but you might want the have them extrapolated.
        use_multiprocessing: bool
          set to false if, for some reason, you don't want to use mp

        Returns
        -------
        an interpolated DataArray

        """
        if levels is None:
            levels = np.array(
                [
                    1000,
                    975,
                    950,
                    925,
                    900,
                    850,
                    800,
                    750,
                    700,
                    650,
                    600,
                    550,
                    500,
                    450,
                    400,
                    300,
                    200,
                    100,
                ]
            )

        zcoord = self._obj['PRESSURE']
        out = self._obj[varname].salem.interpz(
            zcoord,
            levels,
            dim_name='p',
            fill_value=fill_value,
            use_multiprocessing=use_multiprocessing,
        )
        out['p'].attrs['description'] = 'pressure'
        out['p'].attrs['units'] = 'hPa'
        return out


def open_xr_dataset(file: Path | str) -> xr.Dataset:
    """Thin wrapper around xarray's open_dataset.

    This is needed because variables often have not enough georef attrs
    to be understood alone, and datasets tend to loose their attrs with
    operations...

    Returns
    -------
    an xarray Dataset

    """
    # if geotiff, use Salem
    if isinstance(file, str):
        file = Path(file)
    ext = file.suffix.lower()
    if ext in ('.tif', '.tiff'):
        from salem import GeoTiff

        geo = GeoTiff(file)
        # TODO: currently everything is loaded in memory (baaad)
        da = xr.DataArray(
            geo.get_vardata(),
            coords={
                'x': geo.grid.center_grid.x_coord,
                'y': geo.grid.center_grid.y_coord,
            },
            dims=['y', 'x'],
        )
        ds = xr.Dataset()
        ds.attrs['pyproj_srs'] = geo.grid.proj.srs
        ds['data'] = da
        ds['data'].attrs['pyproj_srs'] = geo.grid.proj.srs
        return ds

    # otherwise rely on xarray
    ds = xr.open_dataset(file)

    # did we get the grid? If not no need to go further
    grid = ds.salem.grid

    # add cartesian coords for WRF
    if 'west_east' in ds.dims:
        ds['west_east'] = ds.salem.grid.x_coord
        ds['south_north'] = ds.salem.grid.y_coord

    # add pyproj string everywhere
    ds.attrs['pyproj_srs'] = grid.proj.srs
    for v in ds.data_vars:
        ds[v].attrs['pyproj_srs'] = grid.proj.srs

    return ds


def open_wrf_dataset(file: Path | str, **kwargs) -> xr.Dataset:
    """Use Salem to open a wrf dataset.

    Thin wrapper around xarray's open_dataset to make WRF files a bit better.

    This is needed because variables often have not enough georef attrs
    to be understood alone, and datasets tend to loose their attrs with
    operations...

    Parameters
    ----------
    file : str
        the path to the WRF file
    **kwargs : optional
        Additional arguments passed on to ``xarray.open_dataset``.

    Returns
    -------
    an xarray Dataset

    """
    nc = netCDF4.Dataset(file)
    nc.set_auto_mask(False)

    # Change staggered variables to unstaggered ones
    for vn, v in nc.variables.items():
        if wrftools.Unstaggerer.can_do(v):
            nc.variables[vn] = wrftools.Unstaggerer(v)

    # Check if we can add diagnostic variables to the pot
    for vn in wrftools.var_classes:
        cl = getattr(wrftools, vn)
        if vn not in nc.variables and cl.can_do(nc):
            nc.variables[vn] = cl(nc)

    # trick xarray with our custom netcdf
    ds = xr.open_dataset(NetCDF4DataStore(nc), **kwargs)

    # remove time dimension to lon lat
    try:
        for vn in ['XLONG', 'XLAT']:
            v = ds[vn].isel(Time=0)
            ds[vn] = xr.DataArray(v.values, dims=['south_north', 'west_east'])
    except (ValueError, KeyError):
        pass

    # Convert time (if necessary)
    if 'Time' in ds.dims:
        time = netcdf_time(ds)
        if time is not None:
            ds['Time'] = time
        ds = ds.rename({'Time': 'time'})
    tr = {'Time': 'time', 'XLAT': 'lat', 'XLONG': 'lon', 'XTIME': 'xtime'}
    tr = {k: tr[k] for k in tr if k in ds.variables}
    ds = ds.rename(tr)

    # drop ugly vars
    vns = ['Times', 'XLAT_V', 'XLAT_U', 'XLONG_U', 'XLONG_V']
    vns = [vn for vn in vns if vn in ds.variables]
    try:
        ds = ds.drop_vars(vns)
    except AttributeError:
        ds = ds.drop(vns)

    # add cartesian coords
    ds['west_east'] = ds.salem.grid.x_coord
    ds['south_north'] = ds.salem.grid.y_coord

    # add pyproj string everywhere
    ds.attrs['pyproj_srs'] = ds.salem.grid.proj.srs
    for v in ds.data_vars:
        ds[v].attrs['pyproj_srs'] = ds.salem.grid.proj.srs

    return ds


def is_rotated_proj_working() -> np.bool:
    """Check if the pyproj version is working with rotated projections.

    Returns
    -------
        The check result.

    """
    import pyproj

    srs = (
        '+ellps=WGS84 +proj=ob_tran +o_proj=latlon '
        '+to_meter=0.0174532925199433 +o_lon_p=0.0 +o_lat_p=80.5 '
        '+lon_0=357.5 +no_defs'
    )

    p1 = pyproj.Proj(srs)
    p2 = wgs84

    return np.isclose(
        transform_proj(p1, p2, np.array(-20), np.array(-9)),
        [-22.243473889042903, -0.06328365194179102],
        atol=1e-5,
    ).all()


def open_metum_dataset(
    file: Path | str,
    pole_longitude: float | None = None,
    pole_latitude: float | None = None,
    central_rotated_longitude: float = 0.0,
    **kwargs,
) -> xr.Dataset:
    """Wrapper to Met Office Unified Model files (experimental).

    This is needed because these files are a little messy.

    Parameters
    ----------
    file : str
        the path to the MetUM file
    pole_longitude: optional
        Pole longitude position, in unrotated degrees. Defaults to the one
        found in the file (if found) and errors otherwise.
    pole_latitude: optional
        Pole latitude position, in unrotated degrees. Defaults to the one
        found in the file (if found) and errors otherwise.
    central_rotated_longitude: optional
        Longitude rotation about the new pole, in degrees. Defaults to the one
        found in the file (if found) and 0 otherwise.
    **kwargs : optional
        Additional arguments passed on to ``xarray.open_dataset``.

    Returns
    -------
    an xarray Dataset

    """
    if not is_rotated_proj_working():
        msg = (
            'open_metum_dataset currently does not '
            'work with certain PROJ versions: '
            'https://github.com/pyproj4/pyproj/issues/424'
        )
        raise RuntimeError(msg)

    # open with xarray
    ds = xr.open_dataset(file, **kwargs)

    # Correct for lons
    vn_list = ['grid_longitude_t', 'grid_longitude_uv', 'rlon']
    for vn in vn_list:
        if vn in ds.coords:
            v = ds[vn]
            ds[vn] = v.where(v <= 180, v - 360)

    # get pyproj string
    if pole_longitude is None or pole_latitude is None:
        # search for specific attributes names
        n_lon = 'grid_north_pole_longitude'
        n_lat = 'grid_north_pole_latitude'
        # first in dataset
        pole_longitude = ds.attrs.get(n_lon, None)
        pole_latitude = ds.attrs.get(n_lat, None)
        # then as variable attribute
        if pole_longitude is None or pole_latitude is None:
            for v in ds.variables.values():
                if n_lon in v.attrs:
                    pole_longitude = v.attrs[n_lon]
                if n_lat in v.attrs:
                    pole_latitude = v.attrs[n_lat]
                if pole_longitude is not None and pole_latitude is not None:
                    break
    if pole_longitude is None or pole_latitude is None:
        msg = 'Could not determine pole longitude and/or latitude'
        raise RuntimeError(msg)

    srs = (
        '+ellps=WGS84 +proj=ob_tran +o_proj=latlon '
        '+to_meter=0.0174532925199433 '
        '+o_lon_p={o_lon_p} +o_lat_p={o_lat_p} +lon_0={lon_0} +no_defs'
    )
    params = {
        'o_lon_p': central_rotated_longitude,
        'o_lat_p': pole_latitude,
        'lon_0': 180 + pole_longitude,
    }
    srs = srs.format(**params)

    # add pyproj string everywhere
    ds.attrs['pyproj_srs'] = srs
    for v in ds.data_vars:
        ds[v].attrs['pyproj_srs'] = srs

    return ds


def open_mf_wrf_dataset(
    paths: list[Path] | Path | str,
    chunks: int | dict[str, float] | None = None,
    compat: str = 'no_conflicts',
    preprocess: Callable | None = None,
    *,
    lock: bool | threading.Lock = False,
) -> xr.Dataset:
    """Open multiple WRF files as a single WRF dataset.

    Requires dask to be installed. Note that if your files are sliced by time,
    certain diagnostic variable computed out of accumulated variables (e.g.
    PRCP) won't be available, because not computable lazily.

    This code is adapted from xarray's open_mfdataset function. The xarray
    license is reproduced in the salem/licenses directory.

    Parameters
    ----------
    paths :
        Either a list of Path object or a Path object.
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by chunk
        sizes. In general, these should divide the dimensions of each dataset.
        If int, chunk each dimension by ``chunks`` .
        By default, chunks will be chosen to load entire input files into
        memory at once. This has a major impact on performance: please see
        xarray's full documentation for more details.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
    lock : False, True or threading.Lock, optional
        This argument is passed on to :py:func:`dask.array.from_array`. By
        default, a per-variable lock is used when reading data from netCDF
        files with the netcdf4 and h5netcdf engines to avoid issues with
        concurrent access when using dask's multithreaded backend.

    Returns
    -------
    xarray.Dataset

    """
    if not paths:
        msg = 'no files to open'
        raise OSError(msg)

    if isinstance(paths, Path):
        paths = str(paths)
    if isinstance(paths, str):
        # NOTE: this only works on posix systems
        split = paths.split('/')
        glob = split.pop(-1)
        path = '/'.join(split)
        paths = sorted(Path(path).glob(glob))

    used_lock = lock if lock else netcdf4_python_lock
    try:
        datasets = [
            open_wrf_dataset(p, chunks=chunks or {}, lock=used_lock)
            for p in paths
        ]
    except TypeError as err:
        if 'lock' not in str(err):
            raise
        # New xarray backends
        datasets = [open_wrf_dataset(p, chunks=chunks or {}) for p in paths]

    orig_datasets = datasets

    def ds_closer() -> None:
        for ods in orig_datasets:
            ods.close()

    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    try:
        combined = xr.combine_nested(
            datasets,
            combine_attrs='drop_conflicts',
            concat_dim='time',
            compat=compat,
        )
    except ValueError:
        # Older xarray
        combined = xr.combine_nested(
            datasets, concat_dim='time', compat=compat
        )
    except AttributeError:
        # Even older
        combined = xr.auto_combine(datasets, concat_dim='time', compat=compat)
    combined.attrs = datasets[0].attrs

    try:
        combined.set_close(ds_closer)
    except AttributeError:
        from xarray.backends.api import _MultiFileCloser

        mfc = _MultiFileCloser([ods._file_obj for ods in orig_datasets])
        combined._file_obj = mfc

    # drop accumulated vars if needed (TODO: make this not hard coded)
    vns = ['PRCP', 'PRCP_C', 'PRCP_NC']
    vns = [vn for vn in vns if vn in combined.variables]
    try:
        combined = combined.drop_vars(vns)
    except AttributeError:
        combined = combined.drop(vns)

    return combined