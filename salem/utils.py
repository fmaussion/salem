"""Some useful functions

Copyright: Fabien Maussion, 2014-2015

License: GPLv3+
"""
from __future__ import division
from six.moves.urllib.request import urlretrieve
import zipfile
# Builtins
import os
import shutil
import pickle
# External libs
import pyproj
import numpy as np
import pandas as pd
try:
    import netCDF4
except ImportError:
    pass
try:
    import geopandas as gpd
except ImportError:
    pass

# Locals
from salem import wgs84
from salem import gis
from salem import cache_dir
from salem import python_str
from salem import Grid

# TODO: remove this once we sure that we have all WRF files right
tmp_check_wrf = True

# A series of variables and dimension names that Salem will understand
valid_names = dict()
valid_names['x_dim'] = ['west_east', 'lon', 'longitude', 'longitudes', 'lons',
                        'xlong', 'xlong_m', 'dimlon', 'x', 'lon_3', 'long',
                        'phony_dim_0']
valid_names['y_dim'] = ['south_north', 'lat', 'latitude', 'latitudes', 'lats',
                        'xlat', 'xlat_m', 'dimlat', 'y','lat_3', 'phony_dim_1']
valid_names['z_dim'] = ['levelist','level', 'pressure', 'press', 'zlevel', 'z']
valid_names['t_dim'] = ['time', 'times', 'xtime']

valid_names['lon_var'] = ['lon', 'longitude', 'longitudes', 'lons', 'long']
valid_names['lat_var'] = ['lat', 'latitude', 'latitudes', 'lats']
valid_names['time_var'] = ['time', 'times']

gh_zip = 'https://github.com/fmaussion/salem-sample-data/archive/master.zip'


# Number of pixels in an image with a zoom level of 0.
google_pix = 256
# The equitorial radius of the Earth assuming WGS-84 ellipsoid.
google_earth_radius = 6378137.0


def str_in_list(l1, l2):
    """Check if one element of l1 is in l2 and if yes, returns the name of
    that element.

    Examples
    --------
    >>> print(str_in_list(['lat', 'lon'], ['time', 'times']))
    None
    >>> str_in_list(['Time', 'lat', 'lon'], ['time', 'times'])
    'Time'
    """
    vt = [i for i in l1 if i.lower() in l2]
    if len(vt) > 0:
        return vt[0]
    else:
        return None


def empty_cache():  # pragma: no cover
    """Empty salem's cache directory."""

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)


def cached_path(fpath):
    """Checks if a file is cached and returns the corresponding path."""

    p, ext = os.path.splitext(fpath)

    if ext.lower() == '.p':
        # No need to recached pickled files (this is for possible nested calls)
        return fpath

    if ext.lower() != '.shp':
        raise ValueError('File extension not recognised: {}'.format(ext))

    # Cached directory and file
    cp = os.path.commonprefix([cache_dir, p])
    cp = os.path.join(cache_dir, python_str, os.path.relpath(p, cp))
    ct = '{:d}'.format(int(round(os.path.getmtime(fpath)*1000.)))
    of = os.path.join(cp, ct + '.p')
    if os.path.exists(cp):
        # We have to check if the file changed
        if os.path.exists(of):
            return of
        else:
            # the file has changed
            shutil.rmtree(cp)

    os.makedirs(cp)
    return of


def _download_demo_files():
    """Checks if the demo data is already on the cache and downloads it.

    Currently there's no check to see of the server file has changed: this
    is bad. In the mean time, empty_cache() will ensure that the files are
    up-to-date.
    """

    ofile = os.path.join(cache_dir, 'salem-sample-data.zip')
    odir = os.path.join(cache_dir)
    if not os.path.exists(ofile):  # pragma: no cover
        urlretrieve(gh_zip, ofile)
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(odir)

    out = dict()
    sdir = os.path.join(cache_dir, 'salem-sample-data-master', 'salem-test')
    for root, directories, filenames in os.walk(sdir):
        for filename in filenames:
            out[filename] = os.path.join(root, filename)
    return out


def get_demo_file(fname):
    """Returns the path to the desired demo file."""

    d = _download_demo_files()
    if fname in d:
        return d[fname]
    else:
        return None


def read_shapefile(fpath, cached=False):
    """Reads a shapefile using geopandas."""

    _, ext = os.path.splitext(fpath)
    # TODO: remove this crs stuff when geopandas is uptated (> 0.1.1)
    # https://github.com/geopandas/geopandas/issues/199
    if ext.lower() in ['.shp', '.p']:
        if cached:
            cpath = cached_path(fpath)
            if os.path.exists(cpath):
                with open(cpath, 'rb') as f:
                    pick = pickle.load(f)
                out = pick['gpd']
                out.crs = pick['crs']
            else:
                out = read_shapefile(fpath, cached=False)
                pick = dict(gpd=out, crs=out.crs)
                with open(cpath, 'wb') as f:
                    pickle.dump(pick, f)
        else:
            out = gpd.read_file(fpath)
            out['min_x'] = [g.bounds[0] for g in out.geometry]
            out['max_x'] = [g.bounds[2] for g in out.geometry]
            out['min_y'] = [g.bounds[1] for g in out.geometry]
            out['max_y'] = [g.bounds[3] for g in out.geometry]
    else:
        raise ValueError('File extension not recognised: {}'.format(ext))

    return out


def _wrf_grid(nc):
    """Get the WRF projection out of the file."""

    pargs = dict()
    if hasattr(nc, 'PROJ_ENVI_STRING'):
        # HAR
        dx = nc.GRID_DX
        dy = nc.GRID_DY
        pargs['lat_1'] = nc.PROJ_STANDARD_PAR1
        pargs['lat_2'] = nc.PROJ_STANDARD_PAR2
        pargs['lat_0'] = nc.PROJ_CENTRAL_LAT
        pargs['lon_0'] = nc.PROJ_CENTRAL_LON
        pargs['center_lon'] = nc.PROJ_CENTRAL_LON
        if nc.PROJ_NAME == 'Lambert Conformal Conic':
            proj_id = 1
        else:
            proj_id = 99  # pragma: no cover
    else:
        # Normal WRF file
        cen_lon = nc.CEN_LON
        cen_lat = nc.CEN_LAT
        dx = nc.DX
        dy = nc.DY
        pargs['lat_1'] = nc.TRUELAT1
        pargs['lat_2'] = nc.TRUELAT2
        pargs['lat_0'] = nc.MOAD_CEN_LAT
        pargs['lon_0'] = nc.STAND_LON
        pargs['center_lon'] = nc.CEN_LON
        proj_id = nc.MAP_PROJ

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
        raise NotImplementedError('WRF proj not implemented: ' \
                                  '{}'.format(proj_id))

    proj = gis.check_crs(p4)
    if proj is None:
        raise RuntimeError('WRF proj not understood: {}'.format(p4))

    nx = len(nc.dimensions['west_east'])
    ny = len(nc.dimensions['south_north'])
    if hasattr(nc, 'PROJ_ENVI_STRING'):
        # HAR
        x0 = nc.GRID_X00
        y0 = nc.GRID_Y00
    else:
        # Normal WRF file
        e, n = gis.transform_proj(wgs84, proj, cen_lon, cen_lat)
        x0 = -(nx-1) / 2. * dx + e  # DL corner
        y0 = -(ny-1) / 2. * dy + n  # DL corner
    grid = gis.Grid(nxny=(nx, ny), ll_corner=(x0,y0), dxdy=(dx, dy), proj=proj)


    if tmp_check_wrf:
        #  Temporary asserts
        if 'XLONG' in nc.variables:
            # Normal WRF
            mylon, mylat = grid.ll_coordinates
            reflon = nc.variables['XLONG']
            reflat = nc.variables['XLAT']
            if len(reflon.shape) == 3:
                reflon = reflon[0, :, :]
                reflat = reflat[0, :, :]
            assert np.allclose(reflon, mylon, atol=1e-4)
            assert np.allclose(reflat, mylat, atol=1e-4)
        if 'lon' in nc.variables:
            # HAR
            mylon, mylat = grid.ll_coordinates
            reflon = nc.variables['lon']
            reflat = nc.variables['lat']
            if len(reflon.shape) == 3:
                reflon = reflon[0, :, :]
                reflat = reflat[0, :, :]
            assert np.allclose(reflon, mylon, atol=1e-4)
            assert np.allclose(reflat, mylat, atol=1e-4)

    return grid


def _netcdf_lonlat_grid(nc):
    """Seek for longitude and latitude coordinates."""

    # Do we have some standard names as vaiable?
    vns = nc.variables.keys()
    lon = str_in_list(vns, valid_names['lon_var'])
    lat = str_in_list(vns, valid_names['lat_var'])
    if (lon is None) or (lat is None):
        return None

    # OK, get it
    lon = nc.variables[lon][:]
    lat = nc.variables[lat][:]
    if len(lon.shape) != 1:
        raise RuntimeError('Coordinates not of correct shape')

    # Make the grid
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    args = dict(nxny=(lon.shape[0], lat.shape[0]), proj=wgs84, dxdy=(dx, dy))
    args['corner'] = (lon[0], lat[0])
    return gis.Grid(**args)


def netcdf_grid(nc):
    """Find out if the netcdf file contains a grid that Salem understands."""

    if hasattr(nc, 'MOAD_CEN_LAT') or hasattr(nc, 'PROJ_ENVI_STRING'):
        # WRF and HAR have some special attributes
        return _wrf_grid(nc)
    else:
        # Try out platte carree
        return _netcdf_lonlat_grid(nc)


def netcdf_time(nc):
    """Find out if the netcdf file contains a time that Salem understands."""

    time = None
    vt = str_in_list(nc.variables.keys(), valid_names['time_var'])
    if hasattr(nc, 'TITLE') and 'GEOGRID' in nc.TITLE:
        # geogrid file
        pass
    elif 'DateStrLen' in nc.dimensions:
        # WRF file
        time = []
        for t in nc.variables['Times'][:]:
            time.append(pd.to_datetime(t.tostring().decode(), errors='raise',
                                       format='%Y-%m-%d_%H:%M:%S'))
    elif vt is not None:
        # CF time
        var = nc.variables[vt]
        time = netCDF4.num2date(var[:], var.units)

    return time


def local_mercator_grid(center_ll=None, extent=None, nx=None, ny=None,
                        order='ll'):
    """Local transverse mercator map centered on a specified point."""

    # Make a local proj
    lon, lat = center_ll
    proj_params = dict(proj='tmerc', lat_0=0., lon_0=lon,
                       k=0.9996, x_0=0, y_0=0, datum='WGS84')
    projloc = pyproj.Proj(proj_params)

    # Define a spatial resolution
    xx = extent[0]
    yy = extent[1]
    if ny is None and nx is None:
        ny = 600
        nx = ny * xx / yy
    else:
        if nx is not None:
            ny = nx * yy / xx
        if ny is not None:
            nx = ny * xx / yy
    nx = np.rint(nx)
    ny = np.rint(ny)

    e, n = pyproj.transform(wgs84, projloc, lon, lat)

    if order=='ul':
        corner = (-xx / 2. + e, yy / 2. + n)
        dxdy = (xx / nx, - yy / ny)
    else:
        corner = (-xx / 2. + e, -yy / 2. + n)
        dxdy = (xx / nx, yy / ny)

    return Grid(proj=projloc, corner=corner, nxny=(nx, ny), dxdy=dxdy,
                      pixel_ref='corner')


def googlestatic_mercator_grid(center_ll=None, nx=640, ny=640, zoom=12):
    """Mercator map centered on a specified point as seen by google API"""

    # Make a local proj
    lon, lat = center_ll
    proj_params = dict(proj='merc', datum='WGS84')
    projloc = pyproj.Proj(proj_params)

    # Meter per pixel
    mpix = (2 * np.pi * google_earth_radius) / google_pix / (2**zoom)
    xx = nx * mpix
    yy = ny * mpix

    e, n = pyproj.transform(wgs84, projloc, lon, lat)
    corner = (-xx / 2. + e, yy / 2. + n)
    dxdy = (xx / nx, - yy / ny)

    return Grid(proj=projloc, corner=corner, nxny=(nx, ny), dxdy=dxdy,
                pixel_ref='corner')
