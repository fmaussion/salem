"""Projections and grids

Copyright: Fabien Maussion, 2014-2016

License: GPLv3
"""
# Python 2 stuff
from __future__ import division
from six import string_types

# Builtins
import copy
from functools import partial
from collections import OrderedDict

# External libs
import pyproj
import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
try:
    from shapely.ops import transform as shapely_transform
except ImportError:
    pass
try:
    import geopandas as gpd
except ImportError:
    pass
try:
    from osgeo import osr
    has_gdal = True
except ImportError:
    has_gdal = False

# Locals
from salem import lazy_property, wgs84


def check_crs(crs):
    """Checks if the crs represents a valid grid, projection or ESPG string.

    Examples
    --------
    >>> p = check_crs('+units=m +init=epsg:26915')
    >>> p.srs
    '+units=m +init=epsg:26915 '
    >>> p = check_crs('wrong')
    >>> p is None
    True

    Returns
    -------
    A valid crs if possible, otherwise None
    """
    if isinstance(crs, pyproj.Proj) or isinstance(crs, Grid):
        out = crs
    elif isinstance(crs, dict) or isinstance(crs, string_types):
        try:
            out = pyproj.Proj(crs)
        except RuntimeError:
            out = None
    else:
        out = None
    return out


class Grid(object):
    """A structured grid on a map projection.

    Central class in the library, taking over user concerns about the
    gridded representation of georeferenced data. It adds a level of
    abstraction by defining a new coordinate system.

    A grid requires georeferencing information at instantiation and is
    immutable *in principle*. I didn't implement barriers: if you want to mess
    with it, you can (not recommended). Note that in most cases, users won't
    have to define the grid themselves: most georeferenced datasets contain
    enough metadata for Salem to determine the data's grid automatically.

    A grid is defined by a projection, a reference point in this projection,
    a grid spacing and a number of grid points. The grid can be defined in a
    "upper left" convention (reference point at the top left corner,
    dy negative - always -). This is the python convention, but not that of all
    datasets (for example, the output of the WRF model follows a down-left
    corner convention). Therefore, grids can also be defined in a "lower left"
    corner convention (dy positive). The use of one or the other convention
    depends on the data, so the user should take care of what he is doing.

    The reference points of the grid points might be located at the corner of
    the pixel (upper left corner for example if dy is negative) or the center
    of the pixel (most atmospheric datasets follow this convention). The two
    concepts are truly equivalent and each grid instance gives access to one
    representation or another ("center_grid" and "corner_grid" properties).
    Under the hood, Salem uses the representation it needs to do
    the job by accessing either one or the other of these parameters. The user
    should know which convention he needs for his purposes: some grid
    functions and properties are representation dependant (transform,
    ll_coordinates, ...) while some are not (extent,
    corner_ll_coordinates ...).

    Attributes
    ----------
    proj : pyproj.Proj instance
        defining the grid's map projection
    nx : int
        number of grid points in the x direction
    ny : int
        number of grid points in the y direction
    dx : float
        x grid spacing (always positive)
    dy : float
        y grid spacing (positive if ll_corner, negative if ul_corner)
    x0 : float
        reference point in X proj coordinates
    y0 : float
        reference point in Y proj coordinates
    order : str
        'ul_corner' or 'll_corner' convention
    pixel_ref : str
        'center' or 'corner' convention
    ij_coordinates
    x_coord
    y_coord
    xy_coordinates
    ll_coordinates
    center_grid
    corner_grid
    extent

    """

    def __init__(self, proj=wgs84, nxny=None, dxdy=None, corner=None,
                 ul_corner=None, ll_corner=None, pixel_ref='center'):
        """
        Parameters
        ----------
        proj : pyproj.Proj instance
            defines the grid's map projection. Defaults to 'PlateCarree'
            (wgs84)
        nxny : (int, int)
            (nx, ny) number of grid points
        dxdy : (float, float)
            (dx, dy) grid spacing in proj coordinates
        corner : (float, float)
            (x0, y0) cartesian coordinates (in proj) of the upper left
             or lower left corner, depending on the sign of dy
        ul_corner : (float, float)
            (x0, y0) cartesian coordinates (in proj) of the upper left corner
        ll_corner : (float, float)
            (x0, y0) cartesian coordinates (in proj) of the lower left corner
        pixel_ref : str
            either 'center' or 'corner' (default: 'center'). Tells
            the Grid object where the (x0, y0) is located in the grid point

        Notes
        -----
        The corner, ul_corner and ll_corner parameters are mutually exclusive:
        set one and only one. If pixel_ref is set to 'corner', the ul_corner
        parameter specifies the **corner grid point's** upper left corner
        coordinates.  Equivalently, the ll_corner parameter then specifies the
        **corner grid point's** lower left coordinate.

        Examples
        --------
        >>> g = Grid(nxny=(3, 2), dxdy=(1, 1), ll_corner=(0, 0), proj=wgs84)
        >>> lon, lat = g.ll_coordinates
        >>> lon
        array([[ 0.,  1.,  2.],
               [ 0.,  1.,  2.]])
        >>> lat
        array([[ 0.,  0.,  0.],
               [ 1.,  1.,  1.]])
        >>> lon, lat = g.corner_grid.ll_coordinates
        >>> lon
        array([[-0.5,  0.5,  1.5],
               [-0.5,  0.5,  1.5]])
        >>> lat
        array([[-0.5, -0.5, -0.5],
               [ 0.5,  0.5,  0.5]])
        >>> g.corner_grid == g.center_grid  # the two reprs are equivalent
        True
        """

        # Check for coordinate system
        _proj = check_crs(proj)
        if _proj is None:
            raise ValueError('proj must be of type pyproj.Proj')
        self.proj = _proj

        # Check for shortcut
        if corner is not None:
            if dxdy[1] < 0.:
                ul_corner = corner
            else:
                ll_corner = corner

        # Initialise the rest
        self._check_input(nxny=nxny, dxdy=dxdy,
                          ul_corner=ul_corner,
                          ll_corner=ll_corner,
                          pixel_ref=pixel_ref)

        # Quick'n dirty solution for comparison operator
        self._ckeys = ['x0', 'y0', 'nx', 'ny', 'dx', 'dy', 'order']

    def _check_input(self, **kwargs):
        """See which parameter combination we have and set everything."""

        combi_a = ['nxny', 'dxdy', 'ul_corner']
        combi_b = ['nxny', 'dxdy', 'll_corner']
        if all(kwargs[k] is not None for k in combi_a):
            nx, ny = kwargs['nxny']
            dx, dy = kwargs['dxdy']
            x0, y0 = kwargs['ul_corner']
            if (dx <= 0.) or (dy >= 0.):
                raise ValueError('dxdy and input params not compatible')
            order = 'ul'
        elif all(kwargs[k] is not None for k in combi_b):
            nx, ny = kwargs['nxny']
            dx, dy = kwargs['dxdy']
            x0, y0 = kwargs['ll_corner']
            if (dx <= 0.) or (dy <= 0.):
                raise ValueError('dxdy and input params not compatible')
            order = 'll'
        else:
            raise ValueError('Input params not compatible')

        self.nx = np.int(nx)
        self.ny = np.int(ny)
        if (self.nx <= 0.) or (self.ny <= 0.):
            raise ValueError('nxny not valid')
        self.dx = np.float(dx)
        self.dy = np.float(dy)
        self.x0 = np.float(x0)
        self.y0 = np.float(y0)
        self.order = order
        
        # Check for pixel ref
        self.pixel_ref = kwargs['pixel_ref'].lower()
        if self.pixel_ref not in ['corner', 'center']:
            raise ValueError('pixel_ref not recognized')

    def __eq__(self, other):
        """Two grids are considered equal when their defining coordinates
        and projection are equal.

        Note: equality also means floating point equality, with all the
        problems that come with it.

        (independent of the grid's cornered or centered representation.)
        """

        a = dict((k, self.corner_grid.__dict__[k]) for k in self._ckeys)
        b = dict((k, other.corner_grid.__dict__[k]) for k in self._ckeys)
        p1 = self.corner_grid.proj
        p2 = other.corner_grid.proj
        return (a == b) and proj_is_same(p1, p2)

    def __str__(self):
        """str representation of the grid (useful for caching)."""

        a = OrderedDict((k, self.corner_grid.__dict__[k]) for k in self._ckeys)
        a['proj'] = '+'.join(sorted(self.proj.srs.split('+')))
        return str(a)

    @lazy_property
    def center_grid(self):
        """(Grid instance) representing the grid in center coordinates."""
        
        if self.pixel_ref == 'center':
            return self
        else:
            # shift the grid
            x0y0 = ((self.x0 + self.dx / 2.), (self.y0 + self.dy / 2.))
            args = dict(nxny=(self.nx, self.ny), dxdy=(self.dx, self.dy),
                        proj=self.proj, pixel_ref='center')
            args[self.order + '_corner'] = x0y0
            return Grid(**args)
        
    @lazy_property
    def corner_grid(self):
        """(Grid instance) representing the grid in corner coordinates."""

        if self.pixel_ref == 'corner':
            return self
        else:
            # shift the grid
            x0y0 = ((self.x0 - self.dx / 2.), (self.y0 - self.dy / 2.))
            args = dict(nxny=(self.nx, self.ny), dxdy=(self.dx, self.dy),
                        proj=self.proj, pixel_ref='corner')
            args[self.order + '_corner'] = x0y0
            return Grid(**args)

    @property
    def ij_coordinates(self):
        """Tuple of i, j coordinates of the grid points.

        (dependent of the grid's cornered or centered representation.)
        """

        x = np.arange(self.nx)
        y = np.arange(self.ny)
        return np.meshgrid(x, y)

    @property
    def x_coord(self):
        """x coordinates of the grid points (1D, no mesh)"""

        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def y_coord(self):
        """y coordinates of the grid points (1D, no mesh)"""

        return self.y0 + np.arange(self.ny) * self.dy

    @property
    def xy_coordinates(self):
        """Tuple of x, y coordinates of the grid points.

        (dependent of the grid's cornered or centered representation.)
        """

        return np.meshgrid(self.x_coord, self.y_coord)

    @lazy_property
    def ll_coordinates(self):
        """Tuple of longitudes, latitudes of the grid points.

        (dependent of the grid's cornered or centered representation.)
        """

        x, y = self.xy_coordinates
        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        return transform_proj(self.proj, proj_out, x, y)

    @property
    def xstagg_xy_coordinates(self):
        """Tuple of x, y coordinates of the X staggered grid.

        (independent of the grid's cornered or centered representation.)
        """

        x_s = self.corner_grid.x0 + np.arange(self.nx+1) * self.dx
        y = self.center_grid.y0 + np.arange(self.ny) * self.dy
        return np.meshgrid(x_s, y)

    @property
    def ystagg_xy_coordinates(self):
        """Tuple of x, y coordinates of the Y staggered grid.

        (independent of the grid's cornered or centered representation.)
        """

        x = self.center_grid.x0 + np.arange(self.nx) * self.dx
        y_s = self.corner_grid.y0 + np.arange(self.ny+1) * self.dy
        return np.meshgrid(x, y_s)

    @lazy_property
    def xstagg_ll_coordinates(self):
        """Tuple of longitudes, latitudes of the X staggered grid.

        (independent of the grid's cornered or centered representation.)
        """

        x, y = self.xstagg_xy_coordinates
        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        return transform_proj(self.proj, proj_out, x, y)

    @lazy_property
    def ystagg_ll_coordinates(self):
        """Tuple of longitudes, latitudes of the Y staggered grid.

        (independent of the grid's cornered or centered representation.)
        """

        x, y = self.ystagg_xy_coordinates
        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        return transform_proj(self.proj, proj_out, x, y)

    @lazy_property
    def pixcorner_ll_coordinates(self):
        """Tuple of longitudes, latitudes (dims: ny+1, nx+1) at the corners of
        the grid.

        Useful for graphics.Map essentially

        (independant of the grid's cornered or centered representation.)
        """

        x = self.corner_grid.x0 + np.arange(self.nx+1) * self.dx
        y = self.corner_grid.y0 + np.arange(self.ny+1) * self.dy
        x, y = np.meshgrid(x, y)
        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        return transform_proj(self.proj, proj_out, x, y)

    @lazy_property
    def extent(self):
        """[left, right, bottom, top] boundaries of the grid in the grid's
        projection.

        The boundaries are the pixels leftmost, rightmost, lowermost and
        uppermost corners, meaning that they are independent from the grid's
        representation.
        """

        x = np.array([0, self.nx]) * self.dx + self.corner_grid.x0
        ypoint = [0, self.ny] if self.order == 'll' else [self.ny, 0]
        y = np.array(ypoint) * self.dy + self.corner_grid.y0

        return [x[0], x[1], y[0], y[1]]

    def extent_in_crs(self, crs=wgs84):
        """Get the extent of the grid in a desired crs.

        Parameters
        ----------
        crs : crs
            the target coordinate reference system.

        Returns
        -------
        [left, right, bottom, top] boundaries of the grid.
        """

        # this is not so trivial
        # for optimisation we will transform the boundaries only
        _i = np.hstack([np.arange(self.nx),
                        np.ones(self.ny)*self.nx,
                        np.arange(self.nx),
                        np.zeros(self.ny)]).flatten()
        _j = np.hstack([np.zeros(self.nx),
                        np.arange(self.ny),
                        np.ones(self.nx)*self.ny,
                        np.arange(self.ny)]).flatten()
        _i, _j = self.corner_grid.ij_to_crs(_i, _j, crs=crs)
        return [np.min(_i), np.max(_i), np.min(_j), np.max(_j)]

    def regrid(self, nx=None, ny=None, factor=1):
        """Make a copy of the grid with an updated spatial resolution.

        The keyword parameters are mutually exclusive, because the x/y ratio
        of the grid has to be preserved.

        Parameters
        ----------
        nx : int
            the new number of x pixels
        nx : int
            the new number of y pixels
        factor : int
            multiplication factor (factor=3 will generate a grid with
            a spatial resolution 3 times finer)

        Returns
        -------
        a new Grid object.
        """

        if nx is not None:
            factor = nx / self.nx
        if ny is not None:
            factor = ny / self.ny

        nx = self.nx * factor
        ny = self.ny * factor
        dx = self.dx / factor
        dy = self.dy / factor

        x0 = self.corner_grid.x0
        y0 = self.corner_grid.y0
        args = dict(nxny=(nx, ny), dxdy=(dx, dy),
                    proj=self.proj, pixel_ref='corner')
        args[self.order + '_corner'] = (x0, y0)
        g = Grid(**args)
        if self.pixel_ref == 'center':
            g = g.center_grid
        return g

    def ij_to_crs(self, i, j, crs=None, nearest=False):
        """Converts local i, j to cartesian coordinates in a specified crs

        Parameters
        ----------
        i : array of floats
            the grid coordinates of the point(s) you want to convert
        j : array of floats
            the grid coordinates of the point(s) you want to convert
        crs: crs
             the target crs (default: self.proj)
        nearest: bool
             (for Grid crs only) convert to the nearest grid point

        Returns
        -------
        (x, y) coordinates of the points in the specified crs.
        """

        # Default
        if crs is None:
            crs = self.proj

        # Convert i, j to x, y
        x = i * self.dx + self.x0
        y = j * self.dy + self.y0

        # Convert x, y to crs
        _crs = check_crs(crs)
        if isinstance(_crs, pyproj.Proj):
            ret = transform_proj(self.proj, _crs, x, y)
        elif isinstance(_crs, Grid):
            ret = _crs.transform(x, y, crs=self.proj, nearest=nearest)
        else:
            raise ValueError('crs not understood')
        return ret

    def transform(self, x, y, z=None, crs=wgs84, nearest=False, maskout=False):
        """Converts any coordinates into the local grid.

        Parameters
        ----------
        x : ndarray
            the grid coordinates of the point(s) you want to convert
        y : ndarray
            the grid coordinates of the point(s) you want to convert
        z : None
            ignored (but necessary since some shapes have a z dimension)
        crs : crs
            reference system of x, y. Could be a pyproj.Proj instance or a
            Grid instance. In the latter case (x, y) are actually (i, j).
            (Default: lonlat in wgs84).
        nearest : bool
            set to True if you wish to return the closest i, j coordinates
            instead of subpixel coords.
        maskout : bool
            set to true if you want to mask out the transformed
            coordinates that are not within the grid.

        Returns
        -------
        (i, j) coordinates of the points in the local grid.
        """

        # First to local proj
        _crs = check_crs(crs)
        if isinstance(_crs, pyproj.Proj):
            x, y = transform_proj(_crs, self.proj, x, y)
        elif isinstance(_crs, Grid):
            x, y = _crs.ij_to_crs(x, y, crs=self.proj)
        else:
            raise ValueError('crs not understood')

        # Then to local grid
        x = (np.ma.array(x) - self.x0) / self.dx
        y = (np.ma.array(y) - self.y0) / self.dy

        # See if we need to round
        if nearest:
            f = np.rint if self.pixel_ref == 'center' else np.floor
            x = f(x).astype(np.int)
            y = f(y).astype(np.int)

        # Mask?
        if maskout:
            if self.pixel_ref == 'center':
                mask = ~((x >= -0.5) & (x < self.nx-0.5) &
                         (y >= -0.5) & (y < self.ny-0.5))
            else:
                mask = ~((x >= 0) & (x < self.nx) &
                         (y >= 0) & (y < self.ny))
            x = np.ma.array(x, mask=mask)
            y = np.ma.array(y, mask=mask)

        return x, y

    def map_gridded_data(self, data, grid=None, interp='nearest',
                         ks=3, out=None):
        """Reprojects any structured data onto the local grid.

        The z and time dimensions of the data (if provided) are conserved, but
        the projected data will have the x, y dimensions of the local grid.

        Currently, nearest neighbor, linear, and spline interpolation are
        available. The dtype of the input data is guaranteed to be conserved,
        except for int which will be converted to floats if non nearest
        neighbor interpolation is asked.

        Parameters
        ----------
        data : ndarray
            an ndarray of dimensions 2, 3, or 4, the two last ones being y, x.
        grid : Grid
            a Grid instance matching the data
        interp : str
            'nearest' (default), 'linear', or 'spline'
        ks : int
            Degree of the bivariate spline. Default is 3.
        missing : int
            integer value to attribute to invalid data (for integer data
            only, floats invalids are forced to NaNs)
        out : ndarray
            output array to fill instead of creating a new one (useful for
            overwriting stuffs)

        Returns
        -------
        A projected ndarray of the data.
        """

        if grid is None:
            try:
                grid = data.salem.grid  # try xarray
            except:
                pass

        # Input checks
        if not isinstance(grid, Grid):
            raise ValueError('grid should be a Grid instance')

        try:  # in case someone gave an xarray dataarray
            data = data.values
        except AttributeError:
            pass

        in_shape = data.shape
        ndims = len(in_shape)
        if (ndims < 2) or (ndims > 4):
            raise ValueError('data dimension not accepted')
        if (in_shape[-1] != grid.nx) or (in_shape[-2] != grid.ny):
            raise ValueError('data dimension not compatible')

        interp = interp.lower()

        use_nn = False
        if interp == 'nearest':
            use_nn = True

        # Transform the local grid into the input grid (backwards transform)
        # Work in center grid cause that's what we need
        # TODO: this stage could be optimized when many variables need transfo
        i, j = self.center_grid.ij_coordinates
        oi, oj = grid.center_grid.transform(i, j, crs=self.center_grid,
                                            nearest=use_nn, maskout=False)
        pv = np.nonzero((oi >= 0) & (oi < grid.nx) &
                        (oj >= 0) & (oj < grid.ny))

        # Prepare the output
        if out is not None:
            out_data = np.ma.asarray(out)
        else:
            out_shape = list(in_shape)
            out_shape[-2:] = [self.ny, self.nx]
            if (data.dtype.kind == 'i') and (interp == 'nearest'):
                # We dont do integer arithmetics other than nearest
                out_data = np.ma.masked_all(out_shape, dtype=data.dtype)
            elif data.dtype.kind == 'i':
                out_data = np.ma.masked_all(out_shape, dtype=np.float)
            else:
                out_data = np.ma.masked_all(out_shape, dtype=data.dtype)

        # Spare us the trouble
        if len(pv[0]) == 0:
            return out_data

        i, j, oi, oj = i[pv], j[pv], oi[pv], oj[pv]

        # Interpolate
        if interp == 'nearest':
            out_data[..., j, i] = data[..., oj, oi]
        elif interp == 'linear':
            points = (np.arange(grid.ny), np.arange(grid.nx))
            if ndims == 2:
                f = RegularGridInterpolator(points, data, bounds_error=False)
                out_data[j, i] = f((oj, oi))
            if ndims == 3:
                for dimi, cdata in enumerate(data):
                    f = RegularGridInterpolator(points, cdata,
                                                bounds_error=False)
                    out_data[dimi, j, i] = f((oj, oi))
            if ndims == 4:
                for dimj, cdata in enumerate(data):
                    for dimi, ccdata in enumerate(cdata):
                        f = RegularGridInterpolator(points, ccdata,
                                                    bounds_error=False)
                        out_data[dimj, dimi, j, i] = f((oj, oi))
        elif interp == 'spline':
            px, py = np.arange(grid.ny), np.arange(grid.nx)
            if ndims == 2:
                f = RectBivariateSpline(px, py, data, kx=ks, ky=ks)
                out_data[j, i] = f(oj, oi, grid=False)
            if ndims == 3:
                for dimi, cdata in enumerate(data):
                    f = RectBivariateSpline(px, py, cdata, kx=ks, ky=ks)
                    out_data[dimi, j, i] = f(oj, oi, grid=False)
            if ndims == 4:
                for dimj, cdata in enumerate(data):
                    for dimi, ccdata in enumerate(cdata):
                        f = RectBivariateSpline(px, py, ccdata, kx=ks, ky=ks)
                        out_data[dimj, dimi, j, i] = f(oj, oi, grid=False)
        else:
            msg = 'interpolation not understood: {}'.format(interp)
            raise ValueError(msg)

        return np.ma.masked_invalid(out_data)

    def region_of_interest(self, shape=None, geometry=None, grid=None,
                           corners=None, crs=wgs84, roi=None):
        """Computes a region of interest (ROI).

        A ROI is simply a mask of the same size as the grid.

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
            add the new region_of_interest to a previous one (useful for
            multiple geometries for example)
        """

        # Initial mask
        if roi is not None:
            mask = np.array(roi, dtype=np.int16)
        else:
            mask = np.zeros((self.ny, self.nx), dtype=np.int16)

        # Several cases
        if shape is not None:
            import pandas as pd
            if not isinstance(shape, pd.DataFrame):
                from salem.sio import read_shapefile
                shape = read_shapefile(shape)
            # corner grid is needed for rasterio
            transform_geopandas(shape, to_crs=self.corner_grid)
            import rasterio
            with rasterio.drivers():
                mask = rasterio.features.rasterize(shape.geometry, out=mask)
        if geometry is not None:
            import rasterio
            # corner grid is needed for rasterio
            geom = transform_geometry(geometry, crs=crs,
                                      to_crs=self.corner_grid)
            with rasterio.drivers():
                mask = rasterio.features.rasterize(np.atleast_1d(geom),
                                                   out=mask)
        if grid is not None:
            _tmp = np.ones((grid.ny, grid.nx), dtype=np.int16)
            mask = self.map_gridded_data(_tmp, grid, out=mask).filled(0)
        if corners is not None:
            cgrid = self.center_grid
            xy0, xy1 = corners
            x0, y0 = cgrid.transform(*xy0, crs=crs, nearest=True)
            x1, y1 = cgrid.transform(*xy1, crs=crs, nearest=True)
            mask[np.min([y0, y1]):np.max([y0, y1]) + 1,
            np.min([x0, x1]):np.max([x0, x1]) + 1] = 1

        return mask


def proj_is_same(p1, p2):
    """Checks is two pyproj projections are equal.

    See https://github.com/jswhit/pyproj/issues/15#issuecomment-208862786

    Parameters
    ----------
    p1 : pyproj.Proj
        first projection
    p2 : pyproj.Proj
        second projection
    """
    if has_gdal:
        # this is more robust, but gdal is a pain
        s1 = osr.SpatialReference()
        s1.ImportFromProj4(p1.srs)
        s2 = osr.SpatialReference()
        s2.ImportFromProj4(p2.srs)
        return s1.IsSame(s2)
    else:
        # at least we can try to sort it
        p1 = '+'.join(sorted(p1.srs.split('+')))
        p2 = '+'.join(sorted(p2.srs.split('+')))
        return p1 == p2


def transform_proj(p1, p2, x, y, nocopy=False):
    """Wrapper around the pyproj.transform function.

    Transform points between two coordinate systems defined by the Proj
    instances p1 and p2.

    When two projections are equal, this function avoids quite a bunch of
    useless calculations. See https://github.com/jswhit/pyproj/issues/15

    Parameters
    ----------
    p1 : pyproj.Proj
        projection associated to x and y
    p2 : pyproj.Proj
        projection into which x, y must be transformed
    x : ndarray
        eastings
    y : ndarray
        northings
    nocopy : bool
        in case the two projections are equal, you can use nocopy if you wish
    """

    if proj_is_same(p1, p2):
        if nocopy:
            return x, y
        else:
            return copy.deepcopy(x), copy.deepcopy(y)

    return pyproj.transform(p1, p2, x, y)


def transform_geometry(geom, crs=wgs84, to_crs=wgs84):
    """Reprojects a shapely geometry.

    Parameters
    ----------
    geom : shapely geometry
        the geometry to transform
    crs : crs
        the geometry's crs
    to_crs : crs
        the crs into which the geometry must be transformed

    Returns
    -------
    A reprojected geometry
    """

    from_crs = check_crs(crs)
    to_crs = check_crs(to_crs)

    if isinstance(to_crs, pyproj.Proj) and isinstance(from_crs, pyproj.Proj):
        project = partial(transform_proj, from_crs, to_crs)
    elif isinstance(to_crs, Grid):
        project = partial(to_crs.transform, crs=from_crs)
    else:
        raise NotImplementedError()

    return shapely_transform(project, geom)


def transform_geopandas(gdf, to_crs=wgs84, inplace=True):
    """Reprojects a geopandas dataframe.

    Parameters
    ----------
    gdf : geopandas.DataFrame
        the dataframe to transform (must have a crs attribute)
    to_crs : crs
        the crs into which the dataframe must be transformed
    inplace : bool
        the original dataframe will be overwritten (default)

    Returns
    -------
    A projected dataframe
    """

    from_crs = check_crs(gdf.crs)
    to_crs = check_crs(to_crs)

    if inplace:
        out = gdf
    else:
        out = gdf.copy()

    if isinstance(to_crs, pyproj.Proj) and isinstance(from_crs, pyproj.Proj):
        project = partial(transform_proj, from_crs, to_crs)
    elif isinstance(to_crs, Grid):
        project = partial(to_crs.transform, crs=from_crs)
    else:
        raise NotImplementedError()

    # Do the job and set the new attributes
    result = out.geometry.apply(lambda geom: shapely_transform(project, geom))
    result.__class__ = gpd.GeoSeries
    result.crs = to_crs
    out.geometry = result
    out.crs = to_crs

    return out


def mercator_grid(center_ll=None, extent=None, ny=600, nx=None, order='ll'):
    """Local transverse mercator map centered on a specified point.

    Parameters
    ----------
    center_ll : (float, float)
        tuple of lon, lat coordinates where the map will be centered.
    extent : (float, float)
        tuple of eastings, northings giving the extent (in m) of the map
    ny : int
        number of y grid points wanted to cover the map (default: 600)
    nx : int
        number of x grid points wanted to cover the map (mutually exclusive
        with y)
    order : str
        'll' (lower left) or 'ul' (upper left)

    """

    # Make a local proj
    lon, lat = center_ll
    proj_params = dict(proj='tmerc', lat_0=0., lon_0=lon,
                       k=0.9996, x_0=0, y_0=0, datum='WGS84')
    projloc = pyproj.Proj(proj_params)

    # Define a spatial resolution
    xx = extent[0]
    yy = extent[1]
    if nx is None:
        nx = ny * xx / yy
    else:
        ny = nx * yy / xx

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
    """Mercator map centered on a specified point (google API conventions).

    Mostly useful for google maps.
    """

    # Number of pixels in an image with a zoom level of 0.
    google_pix = 256
    # The equitorial radius of the Earth assuming WGS-84 ellipsoid.
    google_earth_radius = 6378137.0

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
