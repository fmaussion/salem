"""Collection of utilitary tools to generate grids."""
import numpy as np
import pyproj
from salem import gis, wgs84, Grid
from salem import utils

# TODO: remove this once we sure that we have all WRF files right
tmp_check_wrf = True

# Number of pixels in an image with a zoom level of 0.
google_pix = 256
# The equitorial radius of the Earth assuming WGS-84 ellipsoid.
google_earth_radius = 6378137.0


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
        raise NotImplementedError('WRF proj not implemented: '
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
    grid = gis.Grid(nxny=(nx, ny), ll_corner=(x0, y0), dxdy=(dx, dy),
                    proj=proj)

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
    lon = utils.str_in_list(vns, utils.valid_names['lon_var'])
    lat = utils.str_in_list(vns, utils.valid_names['lat_var'])
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