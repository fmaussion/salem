"""
Input output functions (but mostly input)
"""
from __future__ import division

import os
from glob import glob
import pickle
from datetime import datetime
from functools import partial
import warnings

import numpy as np
import netCDF4
import cftime

from salem.utils import memory, cached_shapefile_path
from salem import gis, utils, wgs84, wrftools, proj_to_cartopy

import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4DataStore
from xarray.backends.api import _MultiFileCloser
from xarray.core import dtypes
try:
    from xarray.backends.locks import (NETCDFC_LOCK, HDF5_LOCK, combine_locks)
    NETCDF4_PYTHON_LOCK = combine_locks([NETCDFC_LOCK, HDF5_LOCK])
except ImportError:
    # xarray < v0.11
    from xarray.backends.api import _default_lock as NETCDF4_PYTHON_LOCK
try:
    from xarray.core.pycompat import basestring
except ImportError:
    # latest xarray dropped python2 support, so we can safely assume py3 here
    basestring = str

# Locals
from salem import transform_proj


def read_shapefile(fpath, cached=False):
    """Reads a shapefile using geopandas.

    For convenience, it adds four columns to the dataframe:
    [min_x, max_x, min_y, max_y]

    Because reading a shapefile can take a long time, Salem provides a
    caching utility (cached=True). This will save a pickle of the shapefile
    in the cache directory.
    """

    import geopandas as gpd

    _, ext = os.path.splitext(fpath)

    if ext.lower() in ['.shp', '.p']:
        if cached:
            cpath = cached_shapefile_path(fpath)
            # unpickle if cached, read and pickle if not
            if os.path.exists(cpath):
                with open(cpath, 'rb') as f:
                    out = pickle.load(f)
            else:
                out = read_shapefile(fpath, cached=False)
                with open(cpath, 'wb') as f:
                    pickle.dump(out, f)
        else:
            out = gpd.read_file(fpath)
            out['min_x'] = [g.bounds[0] for g in out.geometry]
            out['max_x'] = [g.bounds[2] for g in out.geometry]
            out['min_y'] = [g.bounds[1] for g in out.geometry]
            out['max_y'] = [g.bounds[3] for g in out.geometry]
    else:
        raise ValueError('File extension not recognised: {}'.format(ext))

    return out


@memory.cache(ignore=['grid'])
def _memory_shapefile_to_grid(shape_cpath, grid=None,
                              nxny=None, pixel_ref=None, x0y0=None, dxdy=None,
                              proj=None):
    """Quick solution using joblib in order to not transform many times the
    same shape (useful for maps).

    Since grid is a complex object, joblib seems to have trouble with it.
    So joblib is checking its cache according to the grid params while the job
    is done with grid.
    """

    shape = read_shapefile(shape_cpath, cached=True)
    e = grid.extent_in_crs(crs=shape.crs)
    p = np.nonzero(~((shape['min_x'].to_numpy() > e[1]) |
                     (shape['max_x'].to_numpy() < e[0]) |
                     (shape['min_y'].to_numpy() > e[3]) |
                     (shape['max_y'].to_numpy() < e[2])))
    shape = shape.iloc[p]
    shape = gis.transform_geopandas(shape, to_crs=grid, inplace=True)
    return shape


def read_shapefile_to_grid(fpath, grid):
    """Same as read_shapefile but directly transformed to a grid.

    The whole thing is cached so that the second call will
    will be much faster.

    Parameters
    ----------
    fpath: path to the file
    grid: the arrival grid
    """

    # ensure it is a cached pickle (copy code smell)
    shape_cpath = cached_shapefile_path(fpath)
    if not os.path.exists(shape_cpath):
        out = read_shapefile(fpath, cached=False)
        with open(shape_cpath, 'wb') as f:
            pickle.dump(out, f)

    return _memory_shapefile_to_grid(shape_cpath, grid=grid,
                                     **grid.to_dict())


# TODO: remove this once we sure that we have all WRF files right
tmp_check_wrf = True


def _wrf_grid_from_dataset(ds):
    """Get the WRF projection out of the file."""

    pargs = dict()
    if hasattr(ds, 'PROJ_ENVI_STRING'):
        # HAR and other TU Berlin files
        dx = ds.GRID_DX
        dy = ds.GRID_DY
        pargs['lat_1'] = ds.PROJ_STANDARD_PAR1
        pargs['lat_2'] = ds.PROJ_STANDARD_PAR2
        pargs['lat_0'] = ds.PROJ_CENTRAL_LAT
        pargs['lon_0'] = ds.PROJ_CENTRAL_LON
        pargs['center_lon'] = ds.PROJ_CENTRAL_LON
        if ds.PROJ_NAME in ['Lambert Conformal Conic',
                            'WRF Lambert Conformal']:
            proj_id = 1
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

    if proj_id == 1:
        # Lambert
        p4 = '+proj=lcc +lat_1={lat_1} +lat_2={lat_2} ' \
             '+lat_0={lat_0} +lon_0={lon_0} ' \
             '+x_0=0 +y_0=0 +a=6370000 +b=6370000'
        p4 = p4.format(**pargs)
    elif proj_id == 2:
        # Polar stereo
        p4 = '+proj=stere +lat_ts={lat_1} +lon_0={lon_0} +lat_0=90.0' \
             '+x_0=0 +y_0=0 +a=6370000 +b=6370000'
        p4 = p4.format(**pargs)
    elif proj_id == 3:
        # Mercator
        p4 = '+proj=merc +lat_ts={lat_1} ' \
             '+lon_0={center_lon} ' \
             '+x_0=0 +y_0=0 +a=6370000 +b=6370000'
        p4 = p4.format(**pargs)
    else:
        raise NotImplementedError('WRF proj not implemented yet: '
                                  '{}'.format(proj_id))

    proj = gis.check_crs(p4)
    if proj is None:
        raise RuntimeError('WRF proj not understood: {}'.format(p4))

    # Here we have to accept xarray and netCDF4 datasets
    try:
        nx = len(ds.dimensions['west_east'])
        ny = len(ds.dimensions['south_north'])
    except AttributeError:
        # maybe an xarray dataset
        nx = ds.dims['west_east']
        ny = ds.dims['south_north']
    if hasattr(ds, 'PROJ_ENVI_STRING'):
        # HAR
        x0 = ds['west_east'][0]
        y0 = ds['south_north'][0]
    else:
        # Normal WRF file
        e, n = gis.transform_proj(wgs84, proj, cen_lon, cen_lat)
        x0 = -(nx-1) / 2. * dx + e  # DL corner
        y0 = -(ny-1) / 2. * dy + n  # DL corner
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
            raise RuntimeError("couldn't test for correct WRF lon-lat")

        if len(reflon.shape) == 3:
            reflon = reflon[0, :, :]
            reflat = reflat[0, :, :]
        mylon, mylat = grid.ll_coordinates

        atol = 5e-3 if proj_id == 2 else 1e-3
        check = np.isclose(reflon, mylon, atol=atol)
        if not np.alltrue(check):
            n_pix = np.sum(~check)
            maxe = np.max(np.abs(reflon - mylon))
            if maxe < (360 - atol):
                warnings.warn('For {} grid points, the expected accuracy ({}) '
                              'of our lons did not match those of the WRF '
                              'file. Max error: {}'.format(n_pix, atol, maxe))
        check = np.isclose(reflat, mylat, atol=atol)
        if not np.alltrue(check):
            n_pix = np.sum(~check)
            maxe = np.max(np.abs(reflat - mylat))
            warnings.warn('For {} grid points, the expected accuracy ({}) '
                          'of our lats did not match those of the WRF file. '
                          'Max error: {}'.format(n_pix, atol, maxe))

    return grid


def _lonlat_grid_from_dataset(ds):
    """Seek for longitude and latitude coordinates."""

    # Do we have some standard names as variable?
    vns = ds.variables.keys()
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
    if not utils.str_in_list([x], utils.valid_names['lon_var']) or \
            not utils.str_in_list([y], utils.valid_names['lat_var']):
        # name not usual. see if at least the range follows some conv
        if (np.max(np.abs(lon)) > 360.1) or (np.max(np.abs(lat)) > 90.1):
            return None

    # Make the grid
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    args = dict(nxny=(lon.shape[0], lat.shape[0]), proj=wgs84, dxdy=(dx, dy),
                x0y0=(lon[0], lat[0]))
    return gis.Grid(**args)


def _salem_grid_from_dataset(ds):
    """Seek for coordinates that Salem might have created.

    Current convention: x_coord, y_coord, pyproj_srs as attribute
    """


    # Projection
    try:
        proj = ds.pyproj_srs
    except AttributeError:
        proj = None
    proj = gis.check_crs(proj)
    if proj is None:
        return None

    # Do we have some standard names as variable?
    vns = ds.variables.keys()
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
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    args = dict(nxny=(x.shape[0], y.shape[0]), proj=proj, dxdy=(dx, dy),
                x0y0=(x[0], y[0]))
    return gis.Grid(**args)


def grid_from_dataset(ds):
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


def netcdf_time(ncobj, monthbegin=False):
    """Check if the netcdf file contains a time that Salem understands."""

    import pandas as pd

    time = None
    try:
        vt = utils.str_in_list(ncobj.variables.keys(),
                               utils.valid_names['time_var'])[0]
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
            time.append(pd.to_datetime(t.tobytes().decode(),
                                       errors='raise',
                                       format='%Y-%m-%d_%H:%M:%S'))
    elif vt is not None:
        # CF time
        var = ncobj.variables[vt]
        try:
            # We want python times because pandas doesn't understand
            # CFtime
            time = cftime.num2date(var[:], var.units,
                                   only_use_cftime_datetimes=False,
                                   only_use_python_datetimes=True)
        except TypeError:
            # Old versions of cftime did return python times when possible
            time = cftime.num2date(var[:], var.units)

        if monthbegin:
            # sometimes monthly data is centered in the month (stupid)
            time = [datetime(t.year, t.month, 1) for t in time]

    return time


class _XarrayAccessorBase(object):
    """Common logic for for both data structures (DataArray and Dataset).

    http://xarray.pydata.org/en/stable/internals.html#extending-xarray
    """

    def __init__(self, xarray_obj):

        self._obj = xarray_obj

        if isinstance(xarray_obj, xr.DataArray):
            xarray_obj = xarray_obj.to_dataset(name='var')
            try:  # maybe there was already some georef
                xarray_obj.attrs['pyproj_srs'] = xarray_obj['var'].pyproj_srs
            except:
                pass

        self.grid = grid_from_dataset(xarray_obj)
        if self.grid is None:
            raise RuntimeError('dataset Grid not understood.')

        dn = xarray_obj.dims.keys()
        self.x_dim = utils.str_in_list(dn, utils.valid_names['x_dim'])[0]
        self.y_dim = utils.str_in_list(dn, utils.valid_names['y_dim'])[0]
        dim = utils.str_in_list(dn, utils.valid_names['t_dim'])
        self.t_dim = dim[0] if dim else None
        dim = utils.str_in_list(dn, utils.valid_names['z_dim'])
        self.z_dim = dim[0] if dim else None

    def subset(self, margin=0, ds=None, **kwargs):
        """subset(self, margin=0, shape=None, geometry=None, grid=None,
                  corners=None, crs=wgs84, roi=None)

        Get a subset of the dataset.

        Accepts all keywords of :py:func:`~Grid.roi`

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
            a mask for the region of interest to subset the dataset onto
        margin : int
            add a margin to the region to subset (can be negative!). Can
            be used a single keyword, too: set_subset(margin=-5) will remove
            five pixels from each boundary of the dataset.
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

        out_ds = self._obj[{self.x_dim : slice(sub_x[0], sub_x[1]+1),
                            self.y_dim : slice(sub_y[0], sub_y[1]+1)}
                          ]
        return out_ds

    def roi(self, ds=None, **kwargs):
        """roi(self, shape=None, geometry=None, grid=None, corners=None,
               crs=wgs84, roi=None, all_touched=False, other=None)

        Make a region of interest (ROI) for the dataset.

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
        coords = {self.y_dim: self._obj[self.y_dim].values,
                  self.x_dim: self._obj[self.x_dim].values}
        mask = xr.DataArray(mask, coords=coords,
                            dims=(self.y_dim, self.x_dim)) 

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

    def get_map(self, **kwargs):
        """Make a salem.Map out of the dataset.

        All keywords are passed to :py:class:salem.Map
        """

        from salem.graphics import Map
        return Map(self.grid, **kwargs)

    def _quick_map(self, obj, ax=None, interp='nearest', **kwargs):
        """Make a plot of a data array."""

        # some metadata?
        title = obj.name or ''
        if obj._title_for_slice():
            title += ' (' + obj._title_for_slice() + ')'
        cb = obj.attrs['units'] if 'units' in obj.attrs else ''

        smap = self.get_map(**kwargs)
        smap.set_data(obj.values, interp=interp)
        smap.visualize(ax=ax, title=title, cbar_title=cb)
        return smap

    def cartopy(self):
        """Get a cartopy.crs.Projection for this dataset."""
        return proj_to_cartopy(self.grid.proj)

    def _apply_transform(self, transform, grid, other, return_lut=False):
        """Common transform mixin"""

        was_dataarray = False
        if not isinstance(other, xr.Dataset):
            try:
                other = other.to_dataset(name=other.name)
                was_dataarray = True
            except AttributeError:
                # must be a ndarray
                if return_lut:
                    rdata, lut = transform(other, grid=grid, return_lut=True)
                else:
                    rdata = transform(other, grid=grid)
                # let's guess
                sh = rdata.shape
                nd = len(sh)
                if nd == 2:
                    dims = (self.y_dim, self.x_dim)
                elif nd == 3:
                    newdim = 'new_dim'
                    if self.t_dim and sh[0] == self._obj.dims[self.t_dim]:
                        newdim = self.t_dim
                    if self.z_dim and sh[0] == self._obj.dims[self.z_dim]:
                        newdim = self.z_dim
                    dims = (newdim, self.y_dim, self.x_dim)
                else:
                    raise NotImplementedError('more than 3 dims not ok yet.')

                coords = {}
                for d in dims:
                    if d in self._obj:
                        coords[d] = self._obj[d]

                out = xr.DataArray(rdata, coords=coords, dims=dims)
                out.attrs['pyproj_srs'] = self.grid.proj.srs
                if return_lut:
                    return out, lut
                else:
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

            rdata = xr.DataArray(rdata, coords=coords, attrs=var.attrs,
                                 dims=dims)
            rdata.attrs['pyproj_srs'] = self.grid.proj.srs
            out[v] = rdata

        if was_dataarray:
            out = out[v]
        else:
            out.attrs['pyproj_srs'] = self.grid.proj.srs

        if return_lut:
            return out, lut
        else:
            return out

    def transform(self, other, grid=None, interp='nearest', ks=3):
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

    def lookup_transform(self, other, grid=None, method=np.mean, lut=None,
                         return_lut=False):
        """Reprojects an other Dataset or DataArray onto self using the
        forward tranform lookup.

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

        transform = partial(self.grid.lookup_transform, method=method, lut=lut)
        return self._apply_transform(transform, grid, other,
                                     return_lut=return_lut)


@xr.register_dataarray_accessor('salem')
class DataArrayAccessor(_XarrayAccessorBase):

    def quick_map(self, ax=None, interp='nearest', **kwargs):
        """Make a plot of the DataArray."""
        return self._quick_map(self._obj, ax=ax, interp=interp, **kwargs)

    def deacc(self, as_rate=True):
        """De-accumulates the variable (i.e. compute the variable's rate).

        The returned variable has one element less over the time dimension.

        The default is to return in units per hour.

        Parameters
        ----------
        as_rate: bool
          set to false if you don't want units per hour,
          but units per given data timestep
        """

        out = self._obj[{self.t_dim : slice(1, len(self._obj[self.t_dim]))}]
        diff = self._obj[{self.t_dim : slice(0, len(self._obj[self.t_dim])-1)}]
        out.values =  out.values - diff.values
        out.attrs['description'] = out.attrs['description'].replace('ACCUMULATED ', '')

        if as_rate:
            dth = self._obj.time[1].values - self._obj.time[0].values
            dth = dth.astype('timedelta64[h]').astype(float)
            out.values = out.values / dth
            out.attrs['units'] += ' h-1'
        else:
            out.attrs['units'] += ' step-1'

        return out

    def interpz(self, zcoord, levels, dim_name='', fill_value=np.NaN,
                use_multiprocessing=True):
        """Interpolates the array along the vertical dimension

        Parameters
        ----------
        zcoord: DataArray
          the z coordinates of the variable. Must be of same dimensions
        levels: 1dArray
          the levels at which to interpolate
        dim_name: str
          the name of the new dimension
        fill_value : np.NaN or 'extrapolate', optional
          how to handle levels below the topography. Default is to mark them
          as invalid, but you might want the have them extrapolated.
        use_multiprocessing: bool
          set to false if, for some reason, you don't want to use mp

        Returns
        -------
        a new DataArray with the interpolated data
        """

        if self.z_dim is None:
            raise RuntimeError('zdimension not recognized')

        data = wrftools.interp3d(self._obj.values, zcoord.values,
                                 np.atleast_1d(levels), fill_value=fill_value,
                                 use_multiprocessing=use_multiprocessing)

        dims = list(self._obj.dims)
        zd = np.nonzero([self.z_dim == d for d in dims])[0][0]
        dims[zd] = dim_name
        coords = dict(self._obj.coords)
        coords.pop(self.z_dim, None)
        coords[dim_name] = np.atleast_1d(levels)
        out = xr.DataArray(data, name=self._obj.name, dims=dims, coords=coords)
        out.attrs['pyproj_srs'] = self.grid.proj.srs
        if not np.asarray(levels).shape:
            out = out.isel(**{dim_name:0})
        return out


@xr.register_dataset_accessor('salem')
class DatasetAccessor(_XarrayAccessorBase):

    def quick_map(self, varname, ax=None, interp='nearest', **kwargs):
        """Make a plot of a variable of the DataSet."""
        return self._quick_map(self._obj[varname], ax=ax, interp=interp,
                               **kwargs)

    def transform_and_add(self, other, grid=None, interp='nearest', ks=3,
                          name=None):
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
                raise ValueError('You need to set name')
            self._obj[new_name] = out
        else:
            for v in out.data_vars:
                try:
                    new_name = name[v]
                except (KeyError, TypeError):
                    new_name = v
                self._obj[new_name] = out[v]

    def wrf_zlevel(self, varname, levels=None, fill_value=np.NaN,
                   use_multiprocessing=True):
        """Interpolates to a specified height above sea level.

        Parameters
        ----------
        varname: str
          the name of the variable to interpolate
        levels: 1d array
          levels at which to interpolate (default: some levels I thought of)
        fill_value : np.NaN or 'extrapolate', optional
          how to handle levels below the topography. Default is to mark them
          as invalid, but you might want the have them extrapolated.
        use_multiprocessing: bool
          set to false if, for some reason, you don't want to use mp

        Returns
        -------
        an interpolated DataArray
        """
        if levels is None:
            levels = np.array([10, 20, 30, 50, 75, 100, 200, 300, 500, 750,
                               1000, 2000, 3000, 5000, 7500, 10000])

        zcoord = self._obj['Z']
        out = self._obj[varname].salem.interpz(zcoord, levels, dim_name='z',
                                               fill_value=fill_value,
                                               use_multiprocessing=
                                               use_multiprocessing)
        out['z'].attrs['description'] = 'height above sea level'
        out['z'].attrs['units'] = 'm'
        return out

    def wrf_plevel(self, varname, levels=None, fill_value=np.NaN,
                   use_multiprocessing=True):
        """Interpolates to a specified pressure level (hPa).

        Parameters
        ----------
        varname: str
          the name of the variable to interpolate
        levels: 1d array
          levels at which to interpolate (default: some levels I thought of)
        fill_value : np.NaN or 'extrapolate', optional
          how to handle levels below the topography. Default is to mark them
          as invalid, but you might want the have them extrapolated.
        use_multiprocessing: bool
          set to false if, for some reason, you don't want to use mp

        Returns
        -------
        an interpolated DataArray
        """
        if levels is None:
            levels = np.array([1000, 975, 950, 925, 900, 850, 800, 750, 700,
                               650, 600, 550, 500, 450, 400, 300, 200, 100])

        zcoord = self._obj['PRESSURE']
        out = self._obj[varname].salem.interpz(zcoord, levels, dim_name='p',
                                               fill_value=fill_value,
                                               use_multiprocessing=
                                               use_multiprocessing)
        out['p'].attrs['description'] = 'pressure'
        out['p'].attrs['units'] = 'hPa'
        return out


def open_xr_dataset(file):
    """Thin wrapper around xarray's open_dataset.

    This is needed because variables often have not enough georef attrs
    to be understood alone, and datasets tend to loose their attrs with
    operations...

    Returns
    -------
    an xarray Dataset
    """

    # if geotiff, use Salem
    p, ext = os.path.splitext(file)
    if (ext.lower() == '.tif') or (ext.lower() == '.tiff'):
        from salem import GeoTiff
        geo = GeoTiff(file)
        # TODO: currently everything is loaded in memory (baaad)
        da = xr.DataArray(geo.get_vardata(),
                          coords={'x': geo.grid.x_coord,
                                  'y': geo.grid.y_coord},
                          dims=['y', 'x'])
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


def open_wrf_dataset(file, **kwargs):
    """Wrapper around xarray's open_dataset to make WRF files a bit better.

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
    for vn in ['XLONG', 'XLAT']:
        try:
            v = ds[vn].isel(Time=0)
            ds[vn] = xr.DataArray(v.values, dims=['south_north', 'west_east'])
        except (ValueError, KeyError):
            pass

    # Convert time (if necessary)
    if 'Time' in ds.dims:
        time = netcdf_time(ds)
        if time is not None:
            ds['Time'] = time
        ds = ds.rename({'Time':'time'})
    tr = {'Time': 'time', 'XLAT': 'lat', 'XLONG': 'lon', 'XTIME': 'xtime'}
    tr = {k: tr[k] for k in tr.keys() if k in ds.variables}
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


def is_rotated_proj_working():

    import pyproj
    srs = ('+ellps=WGS84 +proj=ob_tran +o_proj=latlon '
           '+to_meter=0.0174532925199433 +o_lon_p=0.0 +o_lat_p=80.5 '
           '+lon_0=357.5 +no_defs')

    p1 = pyproj.Proj(srs)
    p2 = wgs84

    return np.isclose(transform_proj(p1, p2, -20, -9),
                      [-22.243473889042903, -0.06328365194179102],
                      atol=1e-5).all()


def open_metum_dataset(file, pole_longitude=None, pole_latitude=None,
                       central_rotated_longitude=0., **kwargs):
    """Wrapper to Met Office Unified Model files (experimental)

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
        raise RuntimeError('open_metum_dataset currently does not '
                           'work with certain PROJ versions: '
                           'https://github.com/pyproj4/pyproj/issues/424')

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
            for k, v in ds.variables.items():
                if n_lon in v.attrs:
                    pole_longitude = v.attrs[n_lon]
                if n_lat in v.attrs:
                    pole_latitude = v.attrs[n_lat]
                if pole_longitude is not None and pole_latitude is not None:
                    break

    srs = ('+ellps=WGS84 +proj=ob_tran +o_proj=latlon '
           '+to_meter=0.0174532925199433 '
           '+o_lon_p={o_lon_p} +o_lat_p={o_lat_p} +lon_0={lon_0} +no_defs')
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


def open_mf_wrf_dataset(paths, chunks=None,  compat='no_conflicts', lock=None,
                        preprocess=None):
    """Open multiple WRF files as a single WRF dataset.

    Requires dask to be installed. Note that if your files are sliced by time,
    certain diagnostic variable computed out of accumulated variables (e.g.
    PRCP) won't be available, because not computable lazily.

    This code is adapted from xarray's open_mfdataset function. The xarray
    license is reproduced in the salem/licenses directory.

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form `path/to/my/files/*.nc` or an
        explicit list of files to open.
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

    if isinstance(paths, basestring):
        paths = sorted(glob(paths))
    if not paths:
        raise IOError('no files to open')

    # TODO: current workaround to dask thread problems
    import dask
    dask.config.set(scheduler='single-threaded')

    if lock is None:
        lock = NETCDF4_PYTHON_LOCK
    datasets = [open_wrf_dataset(p, chunks=chunks or {}, lock=lock)
                for p in paths]
    file_objs = [ds._file_obj for ds in datasets]

    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]
    try:
        combined = xr.combine_nested(datasets, concat_dim='time',
                                     compat=compat)
    except AttributeError:
        combined = xr.auto_combine(datasets, concat_dim='time', compat=compat)
    combined._file_obj = _MultiFileCloser(file_objs)
    combined.attrs = datasets[0].attrs

    # drop accumulated vars if needed (TODO: make this not hard coded)
    vns = ['PRCP', 'PRCP_C', 'PRCP_NC']
    vns = [vn for vn in vns if vn in combined.variables]
    try:
        combined = combined.drop_vars(vns)
    except AttributeError:
        combined = combined.drop(vns)

    return combined
