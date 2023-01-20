"""
Projections and grids
"""
# Python 2 stuff
from __future__ import division
from six import string_types

# Builtins
import copy
import warnings
from functools import partial
from packaging.version import Version

# External libs
import pyproj
import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

try:
    from osgeo import osr
    has_gdal = True
except ImportError:
    has_gdal = False

# Locals
from salem import lazy_property, wgs84

try:
    crs_type = pyproj.crs.CRS
except AttributeError:
    class Dummy():
        pass
    crs_type = Dummy

def check_crs(crs, raise_on_error=False):
    """Checks if the crs represents a valid grid, projection or ESPG string.

    Examples
    --------
    >>> p = check_crs('epsg:26915 +units=m')
    >>> p.srs
    'epsg:26915 +units=m'
    >>> p = check_crs('wrong')
    >>> p is None
    True

    Returns
    -------
    A valid crs if possible, otherwise None
    """

    try:
        crs = crs.salem.grid  # try xarray
    except:
        pass

    if isinstance(crs, string_types):
        # necessary for python 2
        crs = str(crs)

    err1, err2 = None, None

    if isinstance(crs, pyproj.Proj) or isinstance(crs, Grid):
        out = crs
    elif isinstance(crs, crs_type):
        out = pyproj.Proj(crs.to_wkt(), preserve_units=True)
    elif isinstance(crs, dict) or isinstance(crs, string_types):
        if isinstance(crs, string_types):
            # quick fix for https://github.com/pyproj4/pyproj/issues/345
            crs = crs.replace(' ', '').replace('+', ' +')

        # A series of try-catch to handle the (too) many changes in pyproj
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            try:
                out = pyproj.Proj(crs, preserve_units=True)
            except RuntimeError as e:
                err1 = str(e)
                try:
                    out = pyproj.Proj(init=crs, preserve_units=True)
                except RuntimeError as e:
                    err2 = str(e)
                    out = None
    else:
        out = None

    if raise_on_error and out is None:
        msg = ('salem could not properly parse the provided coordinate '
               'reference system (crs). This could be due to errors in your '
               'data, in PyProj, or with salem itself. If this occurs '
               'unexpectedly, report an issue to https://github.com/fmaussion/'
               'salem/issues. Full log: \n'
               'crs: {} ; \n'.format(crs))
        if err1 is not None:
            msg += 'Output of `pyproj.Proj(crs, preserve_units=True)`: {} ; \n'
            msg = msg.format(err1)
        if err2 is not None:
            msg += 'Output of `pyproj.Proj(init=crs, preserve_units=True)`: {}'
            msg = msg.format(err2)
        raise ValueError(msg)

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

    proj
    nx
    ny
    dx
    dy
    x0
    y0
    origin
    pixel_ref
    x_coord
    y_coord
    xy_coordinates
    ll_coordinates
    xstagg_xy_coordinates
    ystagg_xy_coordinates
    xstagg_ll_coordinates
    ystagg_ll_coordinates
    center_grid
    corner_grid
    extent
    """

    def __init__(self, proj=wgs84, nxny=None, dxdy=None, x0y0=None,
                 pixel_ref='center',
                 corner=None, ul_corner=None, ll_corner=None):
        """
        Parameters
        ----------
        proj : pyproj.Proj instance
            defines the grid's map projection. Defaults to 'PlateCarree'
            (wgs84)
        nxny : (int, int)
            (nx, ny) number of grid points
        dxdy : (float, float)
            (dx, dy) grid spacing in proj coordinates. dx must be positive,
            while dy can be positive or negative depending on the origin
            grid point's lecation (upper-left or lower-left)
        x0y0 : (float, float)
            (x0, y0) cartesian coordinates (in proj) of the upper left
            or lower left corner, depending on the sign of dy
        pixel_ref : str
            either 'center' or 'corner' (default: 'center'). Tells
            the Grid object where the (x0, y0) is located in the grid point.
            If ``pixel_ref`` is set to 'corner' and dy < 0, the ``x0y0``
            kwarg specifies the **grid point's upper left** corner
            coordinates.  Equivalently, if dy > 0, x0y0 specifies the
            **grid point's lower left** coordinate.
        corner : (float, float)
            DEPRECATED in favor of ``x0y0``
            (x0, y0) cartesian coordinates (in proj) of the upper left
            or lower left corner, depending on the sign of dy
        ul_corner : (float, float)
            DEPRECATED in favor of ``x0y0``
            (x0, y0) cartesian coordinates (in proj) of the upper left corner
        ll_corner : (float, float)
            DEPRECATED in favor of ``x0y0``
            (x0, y0) cartesian coordinates (in proj) of the lower left corner

        Examples
        --------
        >>> g = Grid(nxny=(3, 2), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
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
        proj = check_crs(proj)
        if proj is None:
            raise ValueError('proj must be of type pyproj.Proj')
        self._proj = proj

        # deprecations
        if corner is not None:
            warnings.warn('The `corner` kwarg is deprecated: '
                          'use `x0y0` instead.', DeprecationWarning)
            x0y0 = corner
        if ul_corner is not None:
            warnings.warn('The `ul_corner` kwarg is deprecated: '
                          'use `x0y0` instead.', DeprecationWarning)
            if dxdy[1] > 0.:
                raise ValueError('dxdy and input params not compatible')
            x0y0 = ul_corner
        if ll_corner is not None:
            warnings.warn('The `ll_corner` kwarg is deprecated: '
                          'use `x0y0` instead.', DeprecationWarning)
            if dxdy[1] < 0.:
                raise ValueError('dxdy and input params not compatible')
            x0y0 = ll_corner

        # Check for shortcut
        if dxdy[1] < 0.:
            ul_corner = x0y0
        else:
            ll_corner = x0y0

        # Initialise the rest
        self._check_input(nxny=nxny, dxdy=dxdy,
                          ul_corner=ul_corner,
                          ll_corner=ll_corner,
                          pixel_ref=pixel_ref)

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
            origin = 'upper-left'
        elif all(kwargs[k] is not None for k in combi_b):
            nx, ny = kwargs['nxny']
            dx, dy = kwargs['dxdy']
            x0, y0 = kwargs['ll_corner']
            if (dx <= 0.) or (dy <= 0.):
                raise ValueError('dxdy and input params not compatible')
            origin = 'lower-left'
        else:
            raise ValueError('Input params not compatible')

        self._nx = int(nx)
        self._ny = int(ny)
        if (self._nx <= 0) or (self._ny <= 0):
            raise ValueError('nxny not valid')
        self._dx = float(dx)
        self._dy = float(dy)
        self._x0 = float(x0)
        self._y0 = float(y0)
        self._origin = origin

        # Check for pixel ref
        self._pixel_ref = kwargs['pixel_ref'].lower()
        if self._pixel_ref not in ['corner', 'center']:
            raise ValueError('pixel_ref not recognized')

    def __eq__(self, other):
        """Two grids are considered equal when their defining coordinates
        and projection are equal.

        Note: equality also means floating point equality, with all the
        problems that come with it.

        (independent of the grid's cornered or centered representation.)
        """

        # Attributes defining the instance
        ckeys = ['x0', 'y0', 'nx', 'ny', 'dx', 'dy', 'origin']

        a = dict((k, getattr(self.corner_grid, k)) for k in ckeys)
        b = dict((k, getattr(other.corner_grid, k)) for k in ckeys)
        p1 = self.corner_grid.proj
        p2 = other.corner_grid.proj
        return (a == b) and proj_is_same(p1, p2)

    def __repr__(self):
        srs = '+'.join(sorted(self.proj.srs.split('+'))).strip()
        summary = ['<salem.Grid>']
        summary += ['  proj: ' + srs]
        summary += ['  pixel_ref: ' + self.pixel_ref]
        summary += ['  origin: ' + str(self.origin)]
        summary += ['  (nx, ny): (' + str(self.nx) + ', ' + str(self.ny) + ')']
        summary += ['  (dx, dy): (' + str(self.dx) + ', ' + str(self.dy) + ')']
        summary += ['  (x0, y0): (' + str(self.x0) + ', ' + str(self.y0) + ')']
        return '\n'.join(summary) + '\n'

    @property
    def proj(self):
        """``pyproj.Proj`` instance defining the grid's map projection."""
        return self._proj

    @property
    def nx(self):
        """number of grid points in the x direction."""
        return self._nx

    @property
    def ny(self):
        """number of grid points in the y direction."""
        return self._ny

    @property
    def dx(self):
        """x grid spacing (always positive)."""
        return self._dx

    @property
    def dy(self):
        """y grid spacing (positive if ll_corner, negative if ul_corner)."""
        return self._dy

    @property
    def x0(self):
        """X reference point in projection coordinates."""
        return self._x0

    @property
    def y0(self):
        """Y reference point in projection coordinates."""
        return self._y0

    @property
    def origin(self):
        """``'upper-left'`` or ``'lower-left'``."""
        return self._origin

    @property
    def pixel_ref(self):
        """if coordinates are at the ``'center'`` or ``'corner'`` of the grid.
        """
        return self._pixel_ref

    @lazy_property
    def center_grid(self):
        """``salem.Grid`` instance representing the grid in center coordinates.
        """

        if self.pixel_ref == 'center':
            return self
        else:
            # shift the grid
            x0y0 = ((self.x0 + self.dx / 2.), (self.y0 + self.dy / 2.))
            args = dict(nxny=(self.nx, self.ny), dxdy=(self.dx, self.dy),
                        proj=self.proj, pixel_ref='center', x0y0=x0y0)
            return Grid(**args)

    @lazy_property
    def corner_grid(self):
        """``salem.Grid`` instance representing the grid in corner coordinates.
        """

        if self.pixel_ref == 'corner':
            return self
        else:
            # shift the grid
            x0y0 = ((self.x0 - self.dx / 2.), (self.y0 - self.dy / 2.))
            args = dict(nxny=(self.nx, self.ny), dxdy=(self.dx, self.dy),
                        proj=self.proj, pixel_ref='corner', x0y0=x0y0)
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
        proj_out = check_crs('EPSG:4326')

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
        proj_out = check_crs('EPSG:4326')
        return transform_proj(self.proj, proj_out, x, y)

    @lazy_property
    def ystagg_ll_coordinates(self):
        """Tuple of longitudes, latitudes of the Y staggered grid.

        (independent of the grid's cornered or centered representation.)
        """

        x, y = self.ystagg_xy_coordinates
        proj_out = check_crs('EPSG:4326')
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
        proj_out = check_crs('EPSG:4326')
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
        ypoint = [0, self.ny] if self.origin == 'lower-left' else [self.ny, 0]
        y = np.array(ypoint) * self.dy + self.corner_grid.y0

        return [x[0], x[1], y[0], y[1]]

    def almost_equal(self, other, rtol=1e-05, atol=1e-08):
        """A less strict comparison between grids.

        Two grids are considered equal when their defining coordinates
        and projection are equal.
        grid1 == grid2 uses floating point equality, which is very strict; here
        we uses numpy's is close instead.

        (independent of the grid's cornered or centered representation.)
        """

        # float attributes defining the instance
        fkeys = ['x0', 'y0', 'dx', 'dy']
        # unambiguous attributes
        ckeys = ['nx', 'ny', 'origin']

        ok = True
        for k in fkeys:
            ok = ok and np.isclose(getattr(self.corner_grid, k),
                                   getattr(other.corner_grid, k),
                                   rtol=rtol, atol=atol)
        for k in ckeys:
            _ok = getattr(self.corner_grid, k) == getattr(other.corner_grid, k)
            ok = ok and _ok
        p1 = self.corner_grid.proj
        p2 = other.corner_grid.proj
        return ok and proj_is_same(p1, p2)

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
        poly = self.extent_as_polygon(crs=crs)
        _i, _j = poly.exterior.xy
        return [np.min(_i), np.max(_i), np.min(_j), np.max(_j)]

    def extent_as_polygon(self, crs=wgs84):
        """Get the extent of the grid in a shapely.Polygon and desired crs.

        Parameters
        ----------
        crs : crs
            the target coordinate reference system.

        Returns
        -------
        [left, right, bottom, top] boundaries of the grid.
        """
        from shapely.geometry import Polygon

        # this is not so trivial
        # for optimisation we will transform the boundaries only
        _i = np.hstack([np.arange(self.nx+1),
                        np.ones(self.ny+1)*self.nx,
                        np.arange(self.nx+1)[::-1],
                        np.zeros(self.ny+1)]).flatten()
        _j = np.hstack([np.zeros(self.nx+1),
                        np.arange(self.ny+1),
                        np.ones(self.nx+1)*self.ny,
                        np.arange(self.ny+1)[::-1]]).flatten()
        _i, _j = self.corner_grid.ij_to_crs(_i, _j, crs=crs)
        return Polygon(zip(_i, _j))

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
        args = dict(nxny=(nx, ny), dxdy=(dx, dy), x0y0=(x0, y0),
                    proj=self.proj, pixel_ref='corner')
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
        try:
            x = i * self.dx + self.x0
            y = j * self.dy + self.y0
        except TypeError:
            x = np.asarray(i) * self.dx + self.x0
            y = np.asarray(j) * self.dy + self.y0

        # Convert x, y to crs
        _crs = check_crs(crs, raise_on_error=True)
        if isinstance(_crs, pyproj.Proj):
            ret = transform_proj(self.proj, _crs, x, y)
        elif isinstance(_crs, Grid):
            ret = _crs.transform(x, y, crs=self.proj, nearest=nearest)
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

        x, y = np.ma.array(x), np.ma.array(y)

        # First to local proj
        _crs = check_crs(crs, raise_on_error=True)
        if isinstance(_crs, pyproj.Proj):
            x, y = transform_proj(_crs, self.proj, x, y)
        elif isinstance(_crs, Grid):
            x, y = _crs.ij_to_crs(x, y, crs=self.proj)

        # Then to local grid
        x = (x - self.x0) / self.dx
        y = (y - self.y0) / self.dy

        # See if we need to round
        if nearest:
            f = np.rint if self.pixel_ref == 'center' else np.floor
            x = f(x).astype(int)
            y = f(y).astype(int)

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

    def grid_lookup(self, other):
        """Performs forward transformation of any other grid into self.

        The principle of forward transform is to obtain, for each grid point of
        ``self`` , all the indices of ``other`` that are located into the
        given grid point. This transformation makes sense ONLY if ``other`` has
        a higher resolution than the object grid. If ``other`` has a similar
        or coarser resolution than ``self`` , choose the more general
        (and much faster) :py:meth:`Grid.map_gridded_data` method.

        Parameters
        ----------
        other : salem.Grid
            the grid that needs to be transformed into self

        Returns
        -------
        a dict: each key (j, i) contains an array of shape (n, 2) where n is
        the number of ``other`` 's grid points found within the grid point
        (j, i)
        """

        # Input checks
        other = check_crs(other)
        if not isinstance(other, Grid):
            raise ValueError('`other` should be a Grid instance')

        # Transform the other grid into the local grid (forward transform)
        # Work in center grid cause that's what we need
        i, j = other.center_grid.ij_coordinates
        i, j = i.flatten(), j.flatten()
        oi, oj = self.center_grid.transform(i, j, crs=other.center_grid,
                                            nearest=True, maskout=True)
        # keep only valid values
        oi, oj, i, j = oi[~oi.mask], oj[~oi.mask], i[~oi.mask], j[~oi.mask]

        out_inds = oi.flatten() + self.nx * oj.flatten()

        # find the links
        ris = np.digitize(out_inds, bins=np.arange(self.nx*self.ny+1))

        # some optim based on the fact that ris has many duplicates
        sort_idx = np.argsort(ris)
        unq_items, unq_count = np.unique(ris[sort_idx], return_counts=True)
        unq_idx = np.split(sort_idx, np.cumsum(unq_count))

        # lets go
        out = dict()
        for idx, ri in zip(unq_idx, unq_items):
            ij = divmod(ri-1, self.nx)
            out[ij] = np.stack((j[idx], i[idx]), axis=1)
        return out

    def lookup_transform(self, data, grid=None, method=np.mean, lut=None,
                         return_lut=False):
        """Performs the forward transformation of gridded data into self.

        This method is suitable when the data grid is of higher resolution
        than ``self``. ``lookup_transform`` performs aggregation of data
        according to a user given rule (e.g. ``np.mean``, ``len``, ``np.std``),
        applied to all grid points found below a grid point in ``self``.


        See also :py:meth:`Grid.grid_lookup` and examples in the docs

        Parameters
        ----------
        data : ndarray
            an ndarray of dimensions 2, 3, or 4, the two last ones being y, x.
        grid : Grid
            a Grid instance matching the data
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
        An aggregated ndarray of the data, in ``self`` coordinates.
        If ``return_lut==True``, also return the lookup table
        """

        # Input checks
        if grid is None:
            grid = check_crs(data)  # xarray
        if not isinstance(grid, Grid):
            raise ValueError('grid should be a Grid instance')
        if hasattr(data, 'values'):
            data = data.values  # xarray

        # dimensional check
        in_shape = data.shape
        ndims = len(in_shape)
        if (ndims < 2) or (ndims > 4):
            raise ValueError('data dimension not accepted')
        if (in_shape[-1] != grid.nx) or (in_shape[-2] != grid.ny):
            raise ValueError('data dimension not compatible')

        if lut is None:
            lut = self.grid_lookup(grid)

        # Prepare the output
        out_shape = list(in_shape)
        out_shape[-2:] = [self.ny, self.nx]

        if data.dtype.kind == 'i':
            out_data = np.zeros(out_shape, dtype=float) * np.NaN
        else:
            out_data = np.zeros(out_shape, dtype=data.dtype) * np.NaN

        def _2d_trafo(ind, outd):
            for ji, l in lut.items():
                outd[ji] = method(ind[l[:, 0], l[:, 1]])
            return outd

        if ndims == 2:
            _2d_trafo(data, out_data)
        if ndims == 3:
            for dimi, cdata in enumerate(data):
                out_data[dimi, ...] = _2d_trafo(cdata, out_data[dimi, ...])
        if ndims == 4:
            for dimj, cdata in enumerate(data):
                for dimi, ccdata in enumerate(cdata):
                    tmp = _2d_trafo(ccdata, out_data[dimj, dimi, ...])
                    out_data[dimj, dimi, ...] = tmp

        # prepare output
        if method is len:
            out_data[~np.isfinite(out_data)] = 0
            out_data = out_data.astype(int)
        else:
            out_data = np.ma.masked_invalid(out_data)

        if return_lut:
            return out_data, lut
        else:
            return out_data

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
        A projected ndarray of the data, in ``self`` coordinates.
        """

        if grid is None:
            try:
                grid = data.salem.grid  # try xarray
            except AttributeError:
                pass

        # Input checks
        if not isinstance(grid, Grid):
            raise ValueError('grid should be a Grid instance')

        try:  # in case someone gave an xarray dataarray
            data = data.values
        except AttributeError:
            pass

        try:  # in case someone gave a masked array (won't work with scipy)
            data = data.filled(np.nan)
        except AttributeError:
            pass

        if data.dtype == np.float32:
            # New in scipy - issue with float32
            data = data.astype(np.float64)

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
                out_data = np.ma.masked_all(out_shape, dtype=float)
            else:
                out_data = np.ma.masked_all(out_shape, dtype=data.dtype)

        # Spare us the trouble
        if len(pv[0]) == 0:
            return out_data

        i, j, oi, oj = i[pv], j[pv], oi[pv], oj[pv]

        # Interpolate
        if interp == 'nearest':
            if out is not None:
                if ndims > 2:
                    raise ValueError('Need 2D for now.')
                vok = np.isfinite(data[oj, oi])
                out_data[j[vok], i[vok]] = data[oj[vok], oi[vok]]
            else:
                out_data[..., j, i] = data[..., oj, oi]
        elif interp == 'linear':
            points = (np.arange(grid.ny), np.arange(grid.nx))
            if ndims == 2:
                f = RegularGridInterpolator(points, data, bounds_error=False)
                if out is not None:
                    tmp = f((oj, oi))
                    vok = np.isfinite(tmp)
                    out_data[j[vok], i[vok]] = tmp[vok]
                else:
                    out_data[j, i] = f((oj, oi))
            if ndims == 3:
                for dimi, cdata in enumerate(data):
                    f = RegularGridInterpolator(points, cdata,
                                                bounds_error=False)
                    if out is not None:
                        tmp = f((oj, oi))
                        vok = np.isfinite(tmp)
                        out_data[dimi, j[vok], i[vok]] = tmp[vok]
                    else:
                        out_data[dimi, j, i] = f((oj, oi))
            if ndims == 4:
                for dimj, cdata in enumerate(data):
                    for dimi, ccdata in enumerate(cdata):
                        f = RegularGridInterpolator(points, ccdata,
                                                    bounds_error=False)
                        if out is not None:
                            tmp = f((oj, oi))
                            vok = np.isfinite(tmp)
                            out_data[dimj, dimi, j[vok], i[vok]] = tmp[vok]
                        else:
                            out_data[dimj, dimi, j, i] = f((oj, oi))
        elif interp == 'spline':
            px, py = np.arange(grid.ny), np.arange(grid.nx)
            if ndims == 2:
                f = RectBivariateSpline(px, py, data, kx=ks, ky=ks)
                if out is not None:
                    tmp = f(oj, oi, grid=False)
                    vok = np.isfinite(tmp)
                    out_data[j[vok], i[vok]] = tmp[vok]
                else:
                    out_data[j, i] = f(oj, oi, grid=False)
            if ndims == 3:
                for dimi, cdata in enumerate(data):
                    f = RectBivariateSpline(px, py, cdata, kx=ks, ky=ks)
                    if out is not None:
                        tmp = f(oj, oi, grid=False)
                        vok = np.isfinite(tmp)
                        out_data[dimi, j[vok], i[vok]] = tmp[vok]
                    else:
                        out_data[dimi, j, i] = f(oj, oi, grid=False)
            if ndims == 4:
                for dimj, cdata in enumerate(data):
                    for dimi, ccdata in enumerate(cdata):
                        f = RectBivariateSpline(px, py, ccdata, kx=ks, ky=ks)
                        if out is not None:
                            tmp = f(oj, oi, grid=False)
                            vok = np.isfinite(tmp)
                            out_data[dimj, dimi, j[vok], i[vok]] = tmp[vok]
                        else:
                            out_data[dimj, dimi, j, i] = f(oj, oi, grid=False)
        else:
            msg = 'interpolation not understood: {}'.format(interp)
            raise ValueError(msg)

        # we have to catch a warning for an unexplained reason
        with warnings.catch_warnings():
            mess = "invalid value encountered in isfinite"
            warnings.filterwarnings("ignore", message=mess)
            out_data = np.ma.masked_invalid(out_data)
        return out_data

    def region_of_interest(self, shape=None, geometry=None, grid=None,
                           corners=None, crs=wgs84, roi=None,
                           all_touched=False):
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
        all_touched : boolean
            pass-through argument for rasterio.features.rasterize, indicating
            that all grid cells which are  clipped by the shapefile defining
            the region of interest should be included (default=False)
        """

        # Initial mask
        if roi is not None:
            mask = np.array(roi, dtype=np.int16)
        else:
            mask = np.zeros((self.ny, self.nx), dtype=np.int16)

        # Collect keyword arguments, overriding anything the user
        # inadvertently added
        rasterize_kws = dict(out=mask, all_touched=all_touched)

        # Several cases
        if shape is not None:
            import pandas as pd
            inplace = False
            if not isinstance(shape, pd.DataFrame):
                from salem.sio import read_shapefile
                shape = read_shapefile(shape)
                inplace = True
            # corner grid is needed for rasterio
            shape = transform_geopandas(shape, to_crs=self.corner_grid,
                                        inplace=inplace)
            import rasterio
            from rasterio.features import rasterize
            with rasterio.Env():
                mask = rasterize(shape.geometry, **rasterize_kws)
        if geometry is not None:
            import rasterio
            from rasterio.features import rasterize
            # corner grid is needed for rasterio
            geom = transform_geometry(geometry, crs=crs,
                                      to_crs=self.corner_grid)
            with rasterio.Env():
                mask = rasterize(np.atleast_1d(geom), **rasterize_kws)
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

    def to_dict(self):
        """Serialize this grid to a dictionary.

        Returns
        -------
        a grid dictionary

        See Also
        --------
        from_dict : create a Grid from a dict
        """
        srs = self.proj.srs
        return dict(proj=self.proj.srs, x0y0=(self.x0, self.y0),
                    nxny=(self.nx, self.ny), dxdy=(self.dx, self.dy),
                    pixel_ref=self.pixel_ref)

    @classmethod
    def from_dict(self, d):
        """Create a Grid from a dictionary

        Parameters
        ----------
        d : dict, required
            the dict with the necessary information

        Returns
        -------
        a salem.Grid instance

        See Also
        --------
        to_dict : create a dict from a Grid
        """
        return Grid(**d)

    def to_json(self, fpath):
        """Serialize this grid to a json file.

        Parameters
        ----------
        fpath : str, required
            the path to the file to create

        See Also
        --------
        from_json : read a json file
        """
        import json
        with open(fpath, 'w') as fp:
            json.dump(self.to_dict(), fp)

    @classmethod
    def from_json(self, fpath):
        """Create a Grid from a json file

        Parameters
        ----------
        fpath : str, required
            the path to the file to open

        Returns
        -------
        a salem.Grid instance

        See Also
        --------
        to_json : create a json file
        """
        import json
        with open(fpath, 'r') as fp:
            d = json.load(fp)
        return Grid.from_dict(d)

    def to_dataset(self):
        """Creates an empty dataset based on the Grid's geolocalisation.

        Returns
        -------
        An xarray.Dataset object ready to be filled with data
        """
        import xarray as xr
        ds = xr.Dataset(coords={'x': (['x', ], self.center_grid.x_coord),
                                'y': (['y', ], self.center_grid.y_coord)}
                        )
        ds.attrs['pyproj_srs'] = self.proj.srs
        return ds

    def to_geometry(self, to_crs=None):
        """Makes a geometrical representation of the grid (e.g. for drawing).

        This can come also handy when doing shape-to-raster operations.

        TODO: currently returns one polygon for each grid point, but this
        could do more.

        Returns
        -------
        a geopandas.GeoDataFrame
        """
        from geopandas import GeoDataFrame
        from shapely.geometry import Polygon
        out = GeoDataFrame()
        geoms = []
        ii = []
        jj = []
        xx = self.corner_grid.x0 + np.arange(self.nx+1) * self.dx
        yy = self.corner_grid.y0 + np.arange(self.ny+1) * self.dy
        for j, (y0, y1) in enumerate(zip(yy[:-1], yy[1:])):
            for i, (x0, x1) in enumerate(zip(xx[:-1], xx[1:])):
                coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
                geoms.append(Polygon(coords))
                jj.append(j)
                ii.append(i)
        out['j'] = jj
        out['i'] = ii
        out['geometry'] = geoms
        out.crs = self.proj.srs

        if check_crs(to_crs):
            transform_geopandas(out, to_crs=to_crs, inplace=True)
        return out


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
        return s1.IsSame(s2) == 1  # IsSame returns 1 or 0
    else:
        # at least we can try to sort it
        p1 = '+'.join(sorted(p1.srs.split('+')))
        p2 = '+'.join(sorted(p2.srs.split('+')))
        return p1 == p2


def _transform_internal(p1, p2, x, y, **kwargs):
    if hasattr(pyproj, 'Transformer'):
        trf = pyproj.Transformer.from_proj(p1, p2, **kwargs)
        return trf.transform(x, y)
    else:
        return pyproj.transform(p1, p2, x, y, **kwargs)


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

    try:
        # This always makes a copy, even if projections are equivalent
        return _transform_internal(p1, p2, x, y, always_xy=True)
    except TypeError:
        if proj_is_same(p1, p2):
            if nocopy:
                return x, y
            else:
                return copy.deepcopy(x), copy.deepcopy(y)

        return _transform_internal(p1, p2, x, y)


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
    elif isinstance(from_crs, Grid):
        project = partial(from_crs.ij_to_crs, crs=to_crs)
    else:
        raise NotImplementedError()

    from shapely.ops import transform
    return transform(project, geom)


def transform_geopandas(gdf, from_crs=None, to_crs=wgs84, inplace=False):
    """Reprojects a geopandas dataframe.

    Parameters
    ----------
    gdf : geopandas.DataFrame
        the dataframe to transform (must have a crs attribute)
    from_crs : crs
        if gdf has no crs attribute (happens when the crs is a salem grid)
    to_crs : crs
        the crs into which the dataframe must be transformed
    inplace : bool
        the original dataframe will be overwritten (default: False)

    Returns
    -------
    A projected dataframe
    """
    from shapely.ops import transform
    import geopandas as gpd

    if from_crs is None:
        from_crs = check_crs(gdf.crs)
    else:
        from_crs = check_crs(from_crs)
    to_crs = check_crs(to_crs)

    if inplace:
        out = gdf
    else:
        out = gdf.copy()

    if isinstance(to_crs, pyproj.Proj) and isinstance(from_crs, pyproj.Proj):
        project = partial(transform_proj, from_crs, to_crs)
    elif isinstance(to_crs, Grid):
        project = partial(to_crs.transform, crs=from_crs)
    elif isinstance(from_crs, Grid):
        project = partial(from_crs.ij_to_crs, crs=to_crs)
    else:
        raise NotImplementedError()

    # Do the job and set the new attributes
    result = out.geometry.apply(lambda geom: transform(project, geom))
    result.__class__ = gpd.GeoSeries
    if isinstance(to_crs, pyproj.Proj):
        to_crs = to_crs.srs
    elif isinstance(to_crs, Grid):
        to_crs = None
    result.crs = to_crs
    out.geometry = result
    out.crs = to_crs
    out['min_x'] = [g.bounds[0] for g in out.geometry]
    out['max_x'] = [g.bounds[2] for g in out.geometry]
    out['min_y'] = [g.bounds[1] for g in out.geometry]
    out['max_y'] = [g.bounds[3] for g in out.geometry]
    return out


def proj_is_latlong(proj):
    """Shortcut function because of deprecation."""

    try:
        return proj.is_latlong()
    except AttributeError:
        return proj.crs.is_geographic


def proj_to_cartopy(proj):
    """Converts a pyproj.Proj to a cartopy.crs.Projection

    Parameters
    ----------
    proj: pyproj.Proj
        the projection to convert

    Returns
    -------
    a cartopy.crs.Projection object

    """

    import cartopy
    import cartopy.crs as ccrs

    proj = check_crs(proj)

    if proj_is_latlong(proj):
        return ccrs.PlateCarree()

    srs = proj.srs
    if has_gdal:
        # this is more robust, as srs could be anything (espg, etc.)
        from osgeo import osr
        s1 = osr.SpatialReference()
        s1.ImportFromProj4(proj.srs)
        if s1.ExportToProj4():
            srs = s1.ExportToProj4()

    km_proj = {'lon_0': 'central_longitude',
               'lat_0': 'central_latitude',
               'x_0': 'false_easting',
               'y_0': 'false_northing',
               'lat_ts': 'latitude_true_scale',
               'o_lon_p': 'central_rotated_longitude',
               'o_lat_p': 'pole_latitude',
               'k': 'scale_factor',
               'zone': 'zone',
               }
    km_globe = {'a': 'semimajor_axis',
                'b': 'semiminor_axis',
                }
    km_std = {'lat_1': 'lat_1',
              'lat_2': 'lat_2',
              }
    kw_proj = dict()
    kw_globe = dict()
    kw_std = dict()
    for s in srs.split('+'):
        s = s.split('=')
        if len(s) != 2:
            continue
        k = s[0].strip()
        v = s[1].strip()
        try:
            v = float(v)
        except:
            pass
        if k == 'proj':
            if v == 'tmerc':
                cl = ccrs.TransverseMercator
                kw_proj['approx'] = True
            if v == 'lcc':
                cl = ccrs.LambertConformal
            if v == 'merc':
                cl = ccrs.Mercator
            if v == 'utm':
                cl = ccrs.UTM
            if v == 'stere':
                cl = ccrs.Stereographic
            if v == 'ob_tran':
                cl = ccrs.RotatedPole
        if k in km_proj:
            if k == 'zone':
                v = int(v)
            kw_proj[km_proj[k]] = v
        if k in km_globe:
            kw_globe[km_globe[k]] = v
        if k in km_std:
            kw_std[km_std[k]] = v

    globe = None
    if kw_globe:
        globe = ccrs.Globe(ellipse='sphere', **kw_globe)
    if kw_std:
        kw_proj['standard_parallels'] = (kw_std['lat_1'], kw_std['lat_2'])

    # mercatoooor
    if cl.__name__ == 'Mercator':
        kw_proj.pop('false_easting', None)
        kw_proj.pop('false_northing', None)
        if Version(cartopy.__version__) < Version('0.15'):
            kw_proj.pop('latitude_true_scale', None)
    elif cl.__name__ == 'Stereographic':
        kw_proj.pop('scale_factor', None)
        if 'latitude_true_scale' in kw_proj:
            kw_proj['true_scale_latitude'] = kw_proj['latitude_true_scale']
            kw_proj.pop('latitude_true_scale', None)
    elif cl.__name__ == 'RotatedPole':
        if 'central_longitude' in kw_proj:
            kw_proj['pole_longitude'] = kw_proj['central_longitude'] - 180
            kw_proj.pop('central_longitude', None)
    else:
        kw_proj.pop('latitude_true_scale', None)

    try:
        return cl(globe=globe, **kw_proj)
    except TypeError:
        del kw_proj['approx']

    return cl(globe=globe, **kw_proj)


def mercator_grid(center_ll=None, extent=None, ny=600, nx=None,
                  origin='lower-left', transverse=True):
    """Local (transverse) mercator map centered on a specified point.

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
    origin : str
        'lower-left' or 'upper-left'
    transverse : bool
        wether to use a transverse or regular mercator. Default should have
        been false, but for backwards compatibility reasons we keep it to True
    """

    # Make a local proj
    pname = 'tmerc' if transverse else 'merc'
    lon, lat = center_ll
    proj_params = dict(proj=pname, lat_0=0., lon_0=lon,
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

    e, n = transform_proj(wgs84, projloc, lon, lat)

    if origin== 'upper-left':
        corner = (-xx / 2. + e, yy / 2. + n)
        dxdy = (xx / nx, - yy / ny)
    else:
        corner = (-xx / 2. + e, -yy / 2. + n)
        dxdy = (xx / nx, yy / ny)

    return Grid(proj=projloc, x0y0=corner, nxny=(nx, ny), dxdy=dxdy,
                pixel_ref='corner')


def googlestatic_mercator_grid(center_ll=None, nx=640, ny=640, zoom=12, scale=1):
    """Mercator map centered on a specified point (google API conventions).

    Mostly useful for google maps.
    """

    # Number of pixels in an image with a zoom level of 0.
    google_pix = 256 * scale
    # The equatorial radius of the Earth assuming WGS-84 ellipsoid.
    google_earth_radius = 6378137.0

    # Make a local proj
    lon, lat = center_ll
    projloc = check_crs('epsg:3857')

    # The size of the image is multiplied by the scaling factor
    nx *= scale
    ny *= scale

    # Meter per pixel
    mpix = (2 * np.pi * google_earth_radius) / google_pix / (2**zoom)
    xx = nx * mpix
    yy = ny * mpix

    e, n = transform_proj(wgs84, projloc, lon, lat)
    corner = (-xx / 2. + e, yy / 2. + n)
    dxdy = (xx / nx, - yy / ny)

    return Grid(proj=projloc, x0y0=corner,
                nxny=(nx, ny), dxdy=dxdy,
                pixel_ref='corner')
