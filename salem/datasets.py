"""This module provides a GeoDataset interface and a few implementations for
e.g. netcdf, geotiff, WRF...

Copyright: Fabien Maussion, 2014-2015

License: GPLv3+
"""
# Builtins
import warnings
import io
import os
from datetime import datetime
from six.moves.urllib.request import urlopen
# External libs
import pyproj
import numpy as np
import salem.grids
import pandas as pd
from matplotlib.image import imread
import xarray as xr
import netCDF4
try:
    import rasterio
    from rasterio import features
except ImportError:
    pass
try:
    import motionless
except ImportError:
    pass

# Locals
from salem import lazy_property
from salem import Grid
from salem import wgs84
from salem import utils
from salem import gis
from salem import wrf


def _to_scalar(x):
    """If a list then scalar"""
    try:
        return x[0]
    except IndexError:
        return x



class GeoDataset(object):
    """Interface for georeferenced datasets,

    A GeoDataset is a formalism for gridded data arrays, which are usually
    stored in geotiffs or netcdf files. It provides an interface to realise
    subsets, compute regions of interest and more.

    A GeoDataset makes more sense if it is subclassed for real files,
    such as GeoTiff or GeoNetCDF. In that case, the implemetations must make
    use of the subset indexes provided in the sub_x, sub_y and sub_t
    properties.
    """

    def __init__(self, grid, time=None):
        """Set-up the georeferencing, time is optional.

        Parameters:
        grid: a salem.Grid object which represents the underlying data
        time: if the data has a time dimension
        """

        # The original grid, for always stored
        self._ogrid = grid
        # The current grid (changes if set_subset() is called)
        self.grid = grid
        # Default indexes to get in the underlying data (BOTH inclusive,
        # i.e [, ], not [,[ as in numpy)
        self.sub_x = [0, grid.nx-1]
        self.sub_y = [0, grid.ny-1]
        # Roi is a ny, nx array if set
        self.roi = None
        self.set_roi()

        # _time is a pd.Series because it's so nice to missuse the series.loc
        # flexibility (see set_period)
        if time is not None:
            if isinstance(time, pd.Series):
                time = pd.Series(np.arange(len(time)), index=time.index)
            else:
                time = pd.Series(np.arange(len(time)), index=time)
        self._time = time

        # set_period() will set those
        self.t0 = None
        self.t1 = None
        self.sub_t = None
        self.set_period()

    @property
    def time(self):
        """Time array"""
        if self._time is None:
            return None
        return self._time[self.t0:self.t1].index

    def set_period(self, t0=[0], t1=[-1]):
        """Set a period of interest for the dataset.

         This will be remembered at later calls to time() or GeoDataset's
         getvardata implementations.

         Parameters
         ----------
         t0: anything that represents a time. Could be a string (e.g
         '2012-01-01'), a DateTime, or an index in the dataset's time
         t1: same as t0 (inclusive)
         """

        if self._time is not None:
            # we dont check for what t0 or t1 is, we let Pandas do the job
            # TODO quick and dirty solution for test_longtime, TBR
            self.sub_t = [_to_scalar(self._time[t0]),
                          _to_scalar(self._time[t1])]
            self.t0 = self._time.index[self.sub_t[0]]
            self.t1 = self._time.index[self.sub_t[1]]

    def set_subset(self, corners=None, crs=wgs84, toroi=False, margin=0):
        """Set a subset for the dataset.

         This will be remembered at later calls to GeoDataset's
         getvardata implementations.

         Parameters
         ----------
         corners: a ((x0, y0), (x1, y1)) tuple of the corners of the square
         to subset the dataset to. The coordinates are not expressed in
         wgs84, set the crs keyword
         crs: the coordinates of the corner coordinates
         toroi: set to true to generate the smallest possible subset arond
         the region of interest set with set_roi()
         margin: when doing the subset, add a margin (can be negative!). Can
         be used alone: set_subset(margin=-5) will remove five pixels from
         each boundary of the dataset.

         TODO: shouldnt we make the toroi stuff easier to use?
         """

        # Useful variables
        mx = self._ogrid.nx-1
        my = self._ogrid.ny-1
        cgrid = self._ogrid.center_grid

        # Three possible cases
        if toroi:
            if self.roi is None or np.max(self.roi) == 0:
                raise RuntimeError('roi is empty.')
            ids = np.nonzero(self.roi)
            sub_x = [np.min(ids[1])-margin, np.max(ids[1])+margin]
            sub_y = [np.min(ids[0])-margin, np.max(ids[0])+margin]
        elif corners is not None:
            xy0, xy1 = corners
            x0, y0 = cgrid.transform(*xy0, crs=crs, nearest=True)
            x1, y1 = cgrid.transform(*xy1, crs=crs, nearest=True)
            sub_x = [np.min([x0, x1])-margin, np.max([x0, x1])+margin]
            sub_y = [np.min([y0, y1])-margin, np.max([y0, y1])+margin]
        else:
            # Reset
            sub_x = [0-margin, mx+margin]
            sub_y = [0-margin, my+margin]

        # Some necessary checks
        if (np.max(sub_x) < 0) or (np.min(sub_x) > mx) or \
           (np.max(sub_y) < 0) or (np.min(sub_y) > my):
            raise RuntimeError('subset not valid')

        if (sub_x[0] < 0) or (sub_x[1] > mx):
            warnings.warn('x0 out of bounds', RuntimeWarning)
        if (sub_y[0] < 0) or (sub_y[1] > my):
            warnings.warn('y0 out of bounds', RuntimeWarning)

        # Make the new grid
        sub_x = np.clip(sub_x, 0, mx)
        sub_y = np.clip(sub_y, 0, my)
        nxny = (sub_x[1] - sub_x[0] + 1, sub_y[1] - sub_y[0] + 1)
        dxdy = (self._ogrid.dx, self._ogrid.dy)
        xy0 = (self._ogrid.x0 + sub_x[0] * self._ogrid.dx,
               self._ogrid.y0 + sub_y[0] * self._ogrid.dy)
        self.grid = Grid(proj=self._ogrid.proj, nxny=nxny, dxdy=dxdy,
                         corner=xy0)
        # If we arrived here, we can safely set the subset
        self.sub_x = sub_x
        self.sub_y = sub_y

    def set_roi(self, shape=None, geometry=None, crs=wgs84, grid=None,
                corners=None, noerase=False):
        """Set a region of interest for the dataset.

        If set succesfully, a ROI is simply a mask of the same size as the
        dataset's grid, obtained with the .roi attribute.

        I haven't decided yet if the data should be masekd out when a ROI
        has been set.

        Parameters
        ----------
        shape: path to a shapefile
        geometry: a shapely geometry
        crs: the crs of the geometry
        grid: a Grid object
        corners: a ((x0, y0), (x1, y1)) tuple of the corners of the square
        to subset the dataset to. The coordinates are not expressed in
        wgs84, set the crs keyword
        noerase: set to true in order to add the new ROI to the previous one
        """

        # The rois are always defined on the original grids, but of course
        # we take that into account when a subset is set (see roi
        # decorator below)
        ogrid = self._ogrid

        # Initial mask
        if noerase and (self.roi is not None):
            mask = self.roi
        else:
            mask = np.zeros((ogrid.ny, ogrid.nx), dtype=np.int16)

        # Several cases
        if shape is not None:
            gdf = utils.read_shapefile(shape)
            gis.transform_geopandas(gdf, to_crs=ogrid.corner_grid)
            with rasterio.drivers():
                mask = features.rasterize(gdf.geometry, out=mask)
        elif geometry is not None:
            geom = gis.transform_geometry(geometry, crs=crs,
                                          to_crs=ogrid.corner_grid)
            with rasterio.drivers():
                mask = features.rasterize(np.atleast_1d(geom), out=mask)
        elif grid is not None:
            _tmp = np.ones((grid.ny, grid.nx), dtype=np.int16)
            mask = ogrid.map_gridded_data(_tmp, grid, out=mask).filled(0)
        elif corners is not None:
            cgrid = self._ogrid.center_grid
            xy0, xy1 = corners
            x0, y0 = cgrid.transform(*xy0, crs=crs, nearest=True)
            x1, y1 = cgrid.transform(*xy1, crs=crs, nearest=True)
            mask[np.min([y0, y1]):np.max([y0, y1])+1,
                 np.min([x0, x1]):np.max([x0, x1])+1] = 1

        self.roi = mask

    @property
    def roi(self):
        """Mask of the ROI (same size as subset)."""
        return self._roi[self.sub_y[0]:self.sub_y[1]+1,
                         self.sub_x[0]:self.sub_x[1]+1]

    @roi.setter
    def roi(self, value):
        """A mask is allways defined on _ogrid"""
        self._roi = value

    def get_vardata(self, var_id=None):
        """Interface to implement by subclasses, taking sub_x, sub_y and
        sub_t into account."""
        raise NotImplementedError()


class GeoTiff(GeoDataset):
    """Open geolocalised tiff images (needs rasterio)."""

    def __init__(self, file):
        """Open the file.

        Parameters
        ----------
        file: path to the file
        """

        # brutally efficient
        with rasterio.drivers():
            with rasterio.open(file) as src:
                nxny = (src.width, src.height)
                ul_corner = (src.bounds.left, src.bounds.top)
                proj = pyproj.Proj(src.crs)
                dxdy = (src.res[0], -src.res[1])
                grid = Grid(ul_corner=ul_corner, nxny=nxny, dxdy=dxdy,
                            pixel_ref='corner', proj=proj)
        # done
        self.file = file
        GeoDataset.__init__(self, grid)

    def get_vardata(self, var_id=1):
        """Read the geotiff band.

        Parameters
        ----------
        var_id: the variable name (here the band number)
        """
        wx = (self.sub_x[0], self.sub_x[1]+1)
        wy = (self.sub_y[0], self.sub_y[1]+1)
        with rasterio.drivers():
            with rasterio.open(self.file) as src:
                band = src.read(var_id, window=(wy, wx))
        return band


class EsriITMIX(GeoDataset):
    """Open ITMIX geolocalised Esri ASCII images (needs rasterio)."""

    def __init__(self, file):
        """Open the file.

        Parameters
        ----------
        file: path to the file
        """

        bname = os.path.basename(file).split('.')[0]
        pok = bname.find('UTM')
        if pok == -1:
            raise ValueError(file + ' does not seem to be an ITMIX file.')
        zone = int(bname[pok+3:])
        south = False
        if zone < 0:
            south = True
            zone = -zone
        proj = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84',
                           south=south)

        # brutally efficient
        with rasterio.drivers():
            with rasterio.open(file) as src:
                nxny = (src.width, src.height)
                ul_corner = (src.bounds.left, src.bounds.top)
                dxdy = (src.res[0], -src.res[1])
                grid = Grid(ul_corner=ul_corner, nxny=nxny, dxdy=dxdy,
                            pixel_ref='corner', proj=proj)
        # done
        self.file = file
        GeoDataset.__init__(self, grid)

    def get_vardata(self, var_id=1):
        """Read the geotiff band.

        Parameters
        ----------
        var_id: the variable name (here the band number)
        """
        wx = (self.sub_x[0], self.sub_x[1]+1)
        wy = (self.sub_y[0], self.sub_y[1]+1)
        with rasterio.drivers():
            with rasterio.open(self.file) as src:
                band = src.read(var_id, window=(wy, wx))
        return band


class GeoNetcdf(GeoDataset):
    """netCDF files with geolocalisation info.

    GeoNetcdf will try hard to understand the geoloc and time of the file,
    but if it can't you can still provide the time and grid at instantiation.
    """

    def __init__(self, file, grid=None, time=None, monthbegin=False):
        """Open the file and try to understand it.

        Parameters
        ----------
        file: path to the netcdf file
        grid: a Grid object. This will override the normal behavior of
        GeoNetcdf, which is to try to understand the grid automatically.
        time: a time array. This will override the normal behavior of
        GeoNetcdf, which is to try to understand the time automatically.
        monthbegin: set to true if you are sure that your data is monthly
        and that the data provider decided to tag the date as the center of
        the month (stupid)
        """

        self._nc = netCDF4.Dataset(file)
        self.variables = self._nc.variables
        if grid is None:
            grid = salem.grids.netcdf_grid(self._nc)
        if time is None:
            time = self._netcdf_time(monthbegin=monthbegin)
        dn = self._nc.dimensions.keys()
        self.t_dim = utils.str_in_list(dn, utils.valid_names['t_dim'])
        self.x_dim = utils.str_in_list(dn, utils.valid_names['x_dim'])
        self.y_dim = utils.str_in_list(dn, utils.valid_names['y_dim'])
        self.z_dim = utils.str_in_list(dn, utils.valid_names['z_dim'])
        GeoDataset.__init__(self, grid, time=time)

    def _netcdf_time(self, monthbegin=False):
        """Check if the netcdf file contains a time that Salem understands."""

        time = None
        vt = utils.str_in_list(self._nc.variables.keys(),
                               utils.valid_names['time_var'])
        if hasattr(self._nc, 'TITLE') and 'GEOGRID' in self._nc.TITLE:
            # geogrid file
            pass
        elif 'DateStrLen' in self._nc.dimensions:
            # WRF file
            time = []
            for t in self._nc.variables['Times'][:]:
                time.append(pd.to_datetime(t.tostring().decode(),
                                           errors='raise',
                                           format='%Y-%m-%d_%H:%M:%S'))
        elif vt is not None:
            # CF time
            var = self._nc.variables[vt]
            time = netCDF4.num2date(var[:], var.units)

            if monthbegin:
                # sometimes monthly data is centered in the month (stupid)
                time = [datetime(t.year, t.month, 1) for t in time]

        return time

    def get_vardata(self, var_id=0, as_xarray=False):
        """Reads the data out of the netCDF file while taking into account
        time and spatial subsets.

        Parameters
        ----------
        var_id: the name of the variable (must be available in self.variables)
        as_xarray: returns a DataArray object
        """

        v = self.variables[var_id]

        # Make the slices
        item = []
        for d in v.dimensions:
            it = slice(None)
            if d == self.t_dim:
                it = slice(self.sub_t[0], self.sub_t[1]+1)
            elif d == self.y_dim:
                it = slice(self.sub_y[0], self.sub_y[1]+1)
            elif d == self.x_dim:
                it = slice(self.sub_x[0], self.sub_x[1]+1)
            item.append(it)

        out = v[tuple(item)]

        if as_xarray:
            # convert to xarray
            dims = v.dimensions
            coords = dict()
            x, y = self.grid.x_coord, self.grid.y_coord
            for d in dims:
                if d == self.t_dim:
                    coords[d] = self.time
                elif d == self.y_dim:
                    coords[d] = y
                elif d == self.x_dim:
                    coords[d] = x
            attrs = v.__dict__.copy()
            bad_keys = ['scale_factor', 'add_offset',
                        '_FillValue', 'missing_value', 'ncvars']
            _ = [attrs.pop(b, None) for b in bad_keys]
            out = xr.DataArray(out, dims=dims, coords=coords, attrs=attrs)

        return out


class WRF(GeoNetcdf):

    def __init__(self, file, grid=None, time=None):

        GeoNetcdf.__init__(self, file, grid=grid, time=time)

        # Change staggered variables to unstaggered ones
        for vn, v in self.variables.items():
            if wrf.Unstaggerer.can_do(v):
                self.variables[vn] = wrf.Unstaggerer(v)

        # Check if we can add diagnostic variables to the pot
        for vn in wrf.var_classes:
            cl = getattr(wrf, vn)
            if cl.can_do(self.variables):
                self.variables[vn] = cl(self.variables)


class GoogleCenterMap(GeoDataset):
    """Download and handle Google Static Maps (needs motionless lib)."""

    def __init__(self, center_ll=(11.38, 47.26), size_x=640, size_y=640,
                 zoom=12, maptype='satellite', use_cache=True, **kwargs):
        """Open the file.

        Parameters
        ----------
        """

        if 'scale' in kwargs:
            raise NotImplementedError('scale not supported')

        # Google grid
        grid = salem.grids.googlestatic_mercator_grid(center_ll=center_ll,
                                                nx=size_x, ny=size_y,
                                                zoom=zoom)

        # Motionless
        googleurl = motionless.CenterMap(lon=center_ll[0], lat=center_ll[1],
                                         size_x=size_x, size_y=size_y,
                                         maptype=maptype, zoom=zoom, **kwargs)

        # done
        self.googleurl = googleurl
        self.use_cache = use_cache
        GeoDataset.__init__(self, grid)

    @lazy_property
    def _img(self):
        """Download the image."""
        if self.use_cache:
            return utils.joblib_read_url(self.googleurl.generate_url())
        else:
            fd = urlopen(self.googleurl.generate_url())
            return imread(io.BytesIO(fd.read()))

    def get_vardata(self, var_id=0):
        """Return and subset the image."""
        return self._img[self.sub_y[0]:self.sub_y[1]+1,
                         self.sub_x[0]:self.sub_x[1]+1, :]


class GoogleVisibleMap(GoogleCenterMap):

    def __init__(self, x, y, src=wgs84, size_x=640, size_y=640, **kwargs):

        if 'zoom' in kwargs:
            raise ValueError('zoom kwarg not accepted.')

        # Transform to lonlat
        src = gis.check_crs(src)
        if isinstance(src, pyproj.Proj):
            lon, lat = gis.transform_proj(src, wgs84, x, y)
        elif isinstance(src, Grid):
            lon, lat = src.ij_to_crs(x, y, crs=wgs84)
        else:
            raise NotImplementedError()

        # surely not the smartest way to do but should be enough for now
        mc = (np.mean(lon), np.mean(lat))
        zoom = 20
        while zoom >= 0:
            grid = salem.grids.googlestatic_mercator_grid(center_ll=mc, nx=size_x,
                                                    ny=size_y, zoom=zoom)
            dx, dy = grid.transform(lon, lat, maskout=True)
            if np.any(dx.mask):
                zoom -= 1
            else:
                break

        GoogleCenterMap.__init__(self, center_ll=mc, size_x=size_x,
                                 size_y=size_y, zoom=zoom, **kwargs)


@xr.register_dataset_accessor('salem')
class XarrayAccessor(GeoDataset):
    """Proof of concept for the new xarray accessors."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.grid = salem.grids.netcdf_grid(self._obj)

        dn = self._obj.dims.keys()
        self.x_dim = utils.str_in_list(dn, utils.valid_names['x_dim'])
        self.y_dim = utils.str_in_list(dn, utils.valid_names['y_dim'])
        GeoDataset.__init__(self, self.grid)

    def subset(self, shape=None, geometry=None, crs=wgs84, grid=None,
                corners=None, margin=0):
        """Get a subset of the dataset.

        Parameters
        ----------
        shape: path to a shapefile
        geometry: a shapely geometry
        crs: the crs of the geometry
        grid: a Grid object
        corners: a ((x0, y0), (x1, y1)) tuple of the corners of the square
        to subset the dataset to. The coordinates are not expressed in
        wgs84, set the crs keyword
        margin: when doing the subset, add a margin (can be negative!). Can
        be used alone: set_subset(margin=-5) will remove five pixels from
        each boundary of the dataset.
        """

        self.set_roi(shape=shape, geometry=geometry, crs=crs, grid=grid,
                     corners=corners)
        self.set_subset(toroi=True, margin=margin)

        out_ds = self._obj[{self.x_dim : slice(self.sub_x[0], self.sub_x[1]+1),
                            self.y_dim : slice(self.sub_y[0], self.sub_y[1]+1)}
                          ]
        self.set_roi()
        self.set_subset()

        return out_ds
