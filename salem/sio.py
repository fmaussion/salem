"""Input output functions (but mostly input)

Copyright: Fabien Maussion, 2014-2016

License: GPLv3+
"""
from __future__ import division

import os
import pickle
import warnings

import numpy as np

from salem.utils import memory, cached_shapefile_path
from salem import gis, utils, wgs84

try:
    import xarray as xr
    has_xarray = True
except ImportError:
    has_xarray = False
    # dummy replacement so that it compiles
    class xr(object):
        pass
    class NullDecl(object):
        def __init__(self, dummy):
            pass
        def __call__(self, func):
            return func
    xr.register_dataset_accessor = NullDecl
    xr.register_dataarray_accessor = NullDecl


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
def _memory_transform(shape_cpath, grid=None, grid_str=None):
    """Quick solution using joblib in order to not transform many times the
    same shape (useful for maps).

    Since grid is a complex object joblib seemed to have trouble with it,
    so joblib is checking its cache according to grid_str while the job is
    done with grid.
    """

    shape = read_shapefile(shape_cpath, cached=True)
    e = grid.extent_in_crs(crs=shape.crs)
    p = np.nonzero(~((shape['min_x'] > e[1]) |
                     (shape['max_x'] < e[0]) |
                     (shape['min_y'] > e[3]) |
                     (shape['max_y'] < e[2])))
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

    return _memory_transform(shape_cpath, grid=grid, grid_str=str(grid))


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
        if ds.PROJ_NAME == 'Lambert Conformal Conic':
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
        x0 = ds.GRID_X00
        y0 = ds.GRID_Y00
    else:
        # Normal WRF file
        e, n = gis.transform_proj(wgs84, proj, cen_lon, cen_lat)
        x0 = -(nx-1) / 2. * dx + e  # DL corner
        y0 = -(ny-1) / 2. * dy + n  # DL corner
    grid = gis.Grid(nxny=(nx, ny), ll_corner=(x0, y0), dxdy=(dx, dy),
                    proj=proj)

    if tmp_check_wrf:
        #  Temporary asserts
        if 'XLONG' in ds.variables:
            # Normal WRF
            mylon, mylat = grid.ll_coordinates
            reflon = ds.variables['XLONG']
            reflat = ds.variables['XLAT']
            if len(reflon.shape) == 3:
                reflon = reflon[0, :, :]
                reflat = reflat[0, :, :]
            assert np.allclose(reflon, mylon, atol=1e-4)
            assert np.allclose(reflat, mylat, atol=1e-4)
        if 'lon' in ds.variables:
            # HAR
            mylon, mylat = grid.ll_coordinates
            reflon = ds.variables['lon']
            reflat = ds.variables['lat']
            if len(reflon.shape) == 3:
                reflon = reflon[0, :, :]
                reflat = reflat[0, :, :]
            assert np.allclose(reflon, mylon, atol=1e-4)
            assert np.allclose(reflat, mylat, atol=1e-4)

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
    args = dict(nxny=(lon.shape[0], lat.shape[0]), proj=wgs84, dxdy=(dx, dy))
    args['corner'] = (lon[0], lat[0])
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
    args = dict(nxny=(x.shape[0], y.shape[0]), proj=proj, dxdy=(dx, dy))
    args['corner'] = (x[0], y[0])
    return gis.Grid(**args)


def grid_from_dataset(ds):
    """Find out if the dataset contains enough info for Salem to understand.

    ds can be an xarray dataset or a NetCDF dataset,
    or anything that resembles it.

    Returns a :py:class:`~salem.Grid` if successful, None otherwise
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


@xr.register_dataset_accessor('salem')
@xr.register_dataarray_accessor('salem')
class XarrayAccessor(object):
    """Accessor for xarray data structures.

    http://xarray.pydata.org/en/stable/internals.html#extending-xarray
    """

    def __init__(self, xarray_obj):

        self._obj = xarray_obj

        if isinstance(xarray_obj, xr.DataArray):
            xarray_obj = xarray_obj.to_dataset(name='var')
            try:
                xarray_obj.attrs['pyproj_srs'] = xarray_obj['var'].pyproj_srs
            except:
                pass

        self.grid = grid_from_dataset(xarray_obj)
        if self.grid is None:
            raise RuntimeError('dataset Grid not understood.')

        dn = xarray_obj.dims.keys()
        self.x_dim = utils.str_in_list(dn, utils.valid_names['x_dim'])[0]
        self.y_dim = utils.str_in_list(dn, utils.valid_names['y_dim'])[0]

    def subset(self, margin=0, **kwargs):
        """Get a subset of the dataset.

        Accepts all keywords of :py:func:`~Grid.roi`

        Parameters
        ----------
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

    def roi(self, **kwargs):
        """Make a region of interest (ROI) for the dataset.

        All grid points outside the ROI will be masked out.

        Parameters
        ----------
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
        """

        mask = self.grid.region_of_interest(**kwargs)
        coords = {self.y_dim: self._obj[self.y_dim].values,
                  self.x_dim: self._obj[self.x_dim].values}
        mask = xr.DataArray(mask, coords=coords,
                            dims=(self.y_dim, self.x_dim))
        out = self._obj.where(mask)

        # keep attrs
        out.attrs = self._obj.attrs
        if isinstance(out, xr.DataArray):
            out.name = self._obj.name

        # add pyproj string everywhere
        out.attrs['pyproj_srs'] = self.grid.proj.srs
        if isinstance(out, xr.Dataset):
            for v in out.variables:
                out.variables[v].attrs = self._obj.variables[v].attrs
                out.variables[v].attrs['pyproj_srs'] = self.grid.proj.srs
        return out

    def get_map(self, **kwargs):
        """Make a salem.Map out of the dataset"""

        from salem.graphics import Map
        return Map(self.grid, **kwargs)

    def plot_on_map(self, name='', ax=None, interp='nearest', **kwargs):
        """Make a plot of the desired variable (or dataarray)."""

        map = self.get_map(**kwargs)

        if isinstance(self._obj, xr.DataArray):
            obj = self._obj
        else:
            obj = self._obj[name]

        title = obj.name or ''
        if obj._title_for_slice():
            title += ' (' + obj._title_for_slice() + ')'

        cb = obj.attrs['units'] if 'units' in obj.attrs else ''

        map.set_data(obj.values, interp=interp)

        map.visualize(ax=ax, title=title, cbar_title=cb)
        return map


def open_xr_dataset(file):
    """Thin wrapper around xarray's open_dataset.

    This is needed because variables often have not enough georef attrs
    to be understood alone, and datasets tend to loose their attrs with
    operations...
    """

    ds = xr.open_dataset(file)

    # did we get it? If not no need to go further
    try:
        grid = ds.salem.grid
    except:
        warnings.warn('File not recognised as Salem grid. Fall back to xarray',
                      RuntimeWarning)
        return ds

    # maybe it's a WRF file?
    if hasattr(ds, 'MOAD_CEN_LAT'):
        # quick n dirty solution for now
        from salem import GeoNetcdf
        with GeoNetcdf(file) as geo:
            ds['Time'] = geo.time
            ds = ds.drop('Times')
            ds.rename({'Time':'time', 'XLAT':'lat', 'XLONG':'lon'},
                      inplace=True)
            ds['west_east'] = geo.grid.x_coord
            ds['south_north'] = geo.grid.y_coord

    # add pyproj string everywhere
    ds.attrs['pyproj_srs'] = grid.proj.srs
    for v in ds.variables:
        ds[v].attrs['pyproj_srs'] = grid.proj.srs

    return ds

