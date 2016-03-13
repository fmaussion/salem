"""Projections and grid information handling.

Salem handles three possible reference coordinate systems (crs). Any point
on Earth can be defined by three parameters: (x, y, crs).
Depending on the crs, x and y represent different quantities::

    DATUM (lon, lat, datum): longitudes and latitudes are angular coordinates
                             of a point on a specified ellipsoid (datum)

    PROJ (x, y, projection): x (eastings) and y (northings) are cartesian
                             coordinates of a point in a specified map
                             projection (unit is _usually_ meter)

    GRID (i, j, grid): on a structured grid, the (x, y) coordinates are
                       distant of a constant (dx, dy) step. The (x, y)
                       coordinates are therefore equivalent to a new
                       reference frame (i, j) proportional to the (x, y)
                       frame of a factor (dx, dy).

The crs DATUM and PROJ are handled by the pyproj library. For simplicity, the
concepts of DATUM and PROJ are interchangeable in pyproj: (lon, lat)
coordinates are equivalent to cartesian (lon, lat) coordinates in the plate
carree projection.

Salem simply adds the concept of GRID to these crs, which I always miss in
other libraries. Grids are very useful when transforming data between two
structured datasets, or from an unstructured dataset to a structured one.

Copyright: Fabien Maussion, 2014-2015

License: GPLv3+
"""
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

# Locals
from salem import lazy_property
from salem import wgs84

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
    """Handling of structured mapped grids.

    Central class in the Salem library, taking over user concerns about the
    gridded representation of georeferenced data. A grid requires four
    parameters at instantiation and is immutable *in principle*.  I didn't
    implement barriers: if you want to mess with it, you can (not recommended).

    A grid is defined by a projection, a reference point in this projection,
    a grid spacing and a number of grid points. The grid can be defined in a
    "upper left" convention (reference point at the top left corner,
    dy negative - always -). This is the python convention, but not that of all
    datasets (for example WRF). It can also be defined in a "lower left" corner
    convention (dy positive). The two conventions are interchangeable only
    if the data that ships with the grid is rotated too, so the user should
    take care of what he is doing.

    The reference points of the grid points might be located at the corner of
    the pixel (upper left corner for example if dy is negative) or the center
    of the pixel (most climate datasets follow this convention). The two
    concepts are truly equivalent and each Grid instance gives access to one
    representation or another with the "center_grid" and "corner_grid"
    properties. Under the hood, Salem uses the representation it needs to do
    the job by accessing either one or the other of these parameters. The user
    in turn should now which convention he needs for his purposes: some grid
    functions and properties are representation dependant (e.g. transform,
    ll_coordinates) while some are not (e,g, extent, corner_ll_coordinates).

    Properties
    ----------
    proj: pyproj.Proj instance
    nx: number of grid points in the x direction
    ny: number of grid points in the y direction
    dx: x grid spacing (always positive)
    dy: y grid spacing (positive if ll_corner, negative if ul_corner)
    x0: reference point in X proj coordinates
    y0: reference point in Y proj coordinates
    order: 'ul_corner' or 'll_corner'
    pixel_ref: 'center' or 'corner'
    ij_coordinates: tuple of two (ny,nx) ndarrays with the i, j coordinates
    of the grid points.
    xy_coordinates: tuple of two (ny,nx) ndarrays with the x, y coordinates
    of the grid points.
    ll_coordinates: tuple of two (ny,nx) ndarrays with the lon,
    lat coordinates of the grid points.
    center_grid: the "pixel centered" representation of the instance
    corner_grid: the "corner centered" representation of the instance
    extent: [left, right, bottom, top] boundaries of the grid in the
    grid's projection.
    xstagg_xy_coordinates, ystagg_xy_coordinates: xy coordinates of the
    staggered grid
    xstagg_ll_coordinates, ystagg_ll_coordinates: ll coordinates of the
    staggered grid
    """

    def __init__(self, proj=wgs84, nxny=None, dxdy=None, corner=None,
                 ul_corner=None, ll_corner=None, pixel_ref='center'):
        """ Instantiate.

        Parameters
        ----------
        proj: pyproj.Proj instance. Defaults to 'PlateCarree' in WGS84
        nxny: (nx, ny) number of grid points
        dxdy: (dx, dy) grid spacing in proj coordinates
        corner: (x0, y0) cartesian coordinates (in proj) of the upper left
        or lower left corner, depending on the sign of dy
        ul_corner: (x0, y0) cartesian coordinates (in proj) of the upper left
        corner.
        ll_corner: (x0, y0) cartesian coordinates (in proj) of the lower left
        corner.
        pixel_ref: either 'center' or 'corner' (default: 'center'). Tells
        the Grid object where the (x0, y0) is located in the grid point

        Comments
        --------
        The ul_corner and ll_corner parameters are mutually exclusive.

        If pixel_ref is set to 'corner', the ul_corner parameter specifies the
        grid point's upper left corner coordinates. Equivalently, the ll_corner
        parameter then specifies the grid point's lower left coordinate.

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
        >>> g.corner_grid == g.center_grid
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
        self._keys = ['x0', 'y0', 'nx', 'ny', 'dx', 'dy', 'order', 'proj']

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

        (independant of the grid's cornered or centered representation.)
        """

        a = dict((k, self.corner_grid.__dict__[k]) for k in self._keys)
        b = dict((k, other.corner_grid.__dict__[k]) for k in self._keys)
        a['proj'] = '+'.join(sorted(a['proj'].srs.split('+')))
        b['proj'] = '+'.join(sorted(b['proj'].srs.split('+')))
        return a == b

    def __str__(self):
        a = OrderedDict((k, self.corner_grid.__dict__[k]) for k in self._keys)
        a['proj'] = '+'.join(sorted(a['proj'].srs.split('+')))
        return str(a)

    @lazy_property
    def center_grid(self):
        """Representation of the grid in center coordinates."""
        
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
        """Representation of the grid in corner coordinates."""

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

        (dependant of the grid's cornered or centered representation.)
        """

        x = np.arange(self.nx)
        y = np.arange(self.ny)
        return np.meshgrid(x, y)

    @property
    def x_coord(self):
        """x coordinates of the grid points (no mesh)"""

        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def y_coord(self):
        """y coordinates of the grid points (no mesh)"""

        return self.y0 + np.arange(self.ny) * self.dy

    @property
    def xy_coordinates(self):
        """Tuple of x, y coordinates of the grid points.

        (dependant of the grid's cornered or centered representation.)
        """

        return np.meshgrid(self.x_coord, self.y_coord)

    @lazy_property
    def ll_coordinates(self):
        """Tuple of longitudes, latitudes of the grid points.

        (dependant of the grid's cornered or centered representation.)
        """

        x, y = self.xy_coordinates
        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        return transform_proj(self.proj, proj_out, x, y)

    @property
    def xstagg_xy_coordinates(self):
        """Tuple of x, y coordinates of the X staggered grid.

        (independant of the grid's cornered or centered representation.)
        """

        x_s = self.corner_grid.x0 + np.arange(self.nx+1) * self.dx
        y = self.center_grid.y0 + np.arange(self.ny) * self.dy
        return np.meshgrid(x_s, y)

    @property
    def ystagg_xy_coordinates(self):
        """Tuple of x, y coordinates of the Y staggered grid.

        (independant of the grid's cornered or centered representation.)
        """

        x = self.center_grid.x0 + np.arange(self.nx) * self.dx
        y_s = self.corner_grid.y0 + np.arange(self.ny+1) * self.dy
        return np.meshgrid(x, y_s)

    @lazy_property
    def xstagg_ll_coordinates(self):
        """Tuple of longitudes, latitudes of the X staggered grid.

        (independant of the grid's cornered or centered representation.)
        """

        x, y = self.xstagg_xy_coordinates
        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        return transform_proj(self.proj, proj_out, x, y)

    @lazy_property
    def ystagg_ll_coordinates(self):
        """Tuple of longitudes, latitudes of the Y staggered grid.

        (independant of the grid's cornered or centered representation.)
        """

        x, y = self.ystagg_xy_coordinates
        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        return transform_proj(self.proj, proj_out, x, y)

    @lazy_property
    def pixcorner_ll_coordinates(self):
        """Tuple of longitudes, latitudes at the corners of the grid.

        Useful for cleo.Map essentially

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
        crs: the target coordinate reference system.

        Returns
        -------
        [left, right, bottom, top] boundaries of the grid.
        """

        # this is not entirely trivial
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

        The keyword parameters are mutually exculive, because the x/y ratio
        of the grid has to be preserved.

        Parameters
        ----------
        nx, ny: the new number of x (y) pixels.
        factor: multiplication factor (factor=3 will generate a grid with
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
        i, j: the grid coordinates of the point(s) you want to convert
        crs: the target crs (default: self.proj)
        nearest: (for Grid crs only) convert to the nearest grid point

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
        x, y: the grid coordinates of the point(s) you want to convert
        z: ignored (but necessary since some shapes hav a z dimension)
        crs: reference system of x, y. Could be a pyproj.Proj instance or a
        Grid instance. In the latter case (x, y) are actually (i, j).
        (Default: lonlat in wgs84).
        nearest: set to True if you wish to return the closest i, j coordinates
        instead of subpixel coords.
        maskout: set to true if you want to mask out the transformed
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
        x = (np.ma.asarray(x) - self.x0) / self.dx
        y = (np.ma.asarray(y) - self.y0) / self.dy

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

    def map_gridded_data(self, data, grid, interp='nearest', ks=3, out=None):
        """Reprojects any structured data onto the local grid.

        The z and time dimensions of the data (if provided) are conserved, but
        the projected data will have the x, y dimensions of the local grid.

        Currently, nearest neighbor and linear interpolation are available.
        the dtype of the input data is guaranteed to be conserved, except for
        int which will be converted to floats if linear interpolation is asked.

        Parameters
        ----------
        data: a ndarray of dimensions 2, 3, or 4, the two last ones being y, x.
        grid: a Grid instance matching the data
        interp: 'nearest' (default), 'linear', or 'spline'
        ks: Degrees of the bivariate spline. Default is 3.
        missing: integer value to attribute to invalid data (for integer data
        only, floats invalids are forced to NaNs)
        out: output array to fill instead of creating a new one (useful for
        overwriting stuffs)

        Returns
        -------
        A projected ndarray of the data.
        """

        # Input checks
        if not isinstance(grid, Grid):
            raise ValueError('grid should be a Grid instance')

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


def transform_proj(p1, p2, x, y, nocopy=False):
    """Wrapper around the pyproj transform.

    When two projections are equal, this function avoids quite a bunch of
    useless calculations. See https://github.com/jswhit/pyproj/issues/15
    """

    if p1.srs == p2.srs:
        if nocopy:
            return x, y
        else:
            return copy.deepcopy(x), copy.deepcopy(y)

    return pyproj.transform(p1, p2, x, y)


def transform_geometry(geom, crs=wgs84, to_crs=wgs84):
    """Reprojects a shapely geometry.

    Parameters
    ----------
    geom: a shapely geometry
    crs: the geometry's crs
    to_crs: the desired crs

    Returns
    -------
    A projected geometry
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
    gdf: geopandas dataframe (must have a crs attribute)
    to_crs: the desired crs

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
