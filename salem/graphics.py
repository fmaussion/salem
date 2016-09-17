"""Color handling and maps.

Copyright: Fabien Maussion, 2014-2016

License: GPLv3+
"""
from __future__ import division
import six

# Builtins
import warnings
import os
from os import path
import copy
# External libs
import numpy as np
from numpy import ma


try:
    from scipy.misc import imresize
except ImportError:
    pass

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.collections import PatchCollection, LineCollection
    from shapely.geometry import MultiPoint
    from descartes.patch import PolygonPatch
except ImportError:
    class d1():
        def __init__(self):
            class d2():
                pass
            self.colors = d2
            self.colors.BoundaryNorm = object
    mpl = d1()

from salem import utils, gis, sio, Grid, wgs84, cache_dir, GeoTiff

# Path to the file directory
file_dir = path.join(cache_dir, 'salem-sample-data-master')
shapefiles = dict()
shapefiles['world_borders'] = path.join(file_dir, 'shapes', 'world_borders',
                                        'world_borders.shp')
shapefiles['oceans'] = path.join(file_dir, 'shapes', 'oceans',
                                 'ne_50m_ocean.shp')
shapefiles['rivers'] = path.join(file_dir, 'shapes', 'rivers',
                                 'ne_50m_rivers_lake_centerlines.shp')

# Be sure we have the directory
if ~ os.path.exists(shapefiles['world_borders']):
    from salem.utils import get_demo_file
    _ = get_demo_file('world_borders.shp')


class ExtendedNorm(mpl.colors.BoundaryNorm):
    """ A better BoundaryNorm with an ``extend'' keyword.

    TODO: remove this when PR is accepted

    See: https://github.com/matplotlib/matplotlib/issues/4850
         https://github.com/matplotlib/matplotlib/pull/5034
    """

    def __init__(self, boundaries, ncolors, extend='neither'):

        _b = np.atleast_1d(boundaries).astype(float)
        mpl.colors.BoundaryNorm.__init__(self, _b, ncolors, clip=False)

        # 'neither' | 'both' | 'min' | 'max'
        if extend == 'both':
            _b = np.append(_b, _b[-1]+1)
            _b = np.insert(_b, 0, _b[0]-1)
        elif extend == 'min':
            _b = np.insert(_b, 0, _b[0]-1)
        elif extend == 'max':
            _b = np.append(_b, _b[-1]+1)
        self._b = _b
        self._N = len(self._b)
        if self._N - 1 == self.Ncmap:
            self._interp = False
        else:
            self._interp = True

    def __call__(self, value):
        xx, is_scalar = self.process_value(value)
        mask = ma.getmaskarray(xx)
        xx = np.atleast_1d(xx.filled(self.vmax + 1))
        iret = np.zeros(xx.shape, dtype=np.int16)
        for i, b in enumerate(self._b):
            iret[xx >= b] = i
        if self._interp:
            scalefac = float(self.Ncmap - 1) / (self._N - 2)
            iret = (iret * scalefac).astype(np.int16)
        iret[xx < self.vmin] = -1
        iret[xx >= self.vmax] = self.Ncmap
        ret = ma.array(iret, mask=mask)
        if is_scalar:
            ret = int(ret[0])  # assume python scalar
        return ret


def get_cm(name='none'):
    """Get a colormap defined by Cleo (more to come!)"""

    cl = utils.read_colormap(name)
    return LinearSegmentedColormap.from_list(name, cl, N=256)


class DataLevels(object):
    """Object to assist you in associating the right color to your data.

    It is a simple object that ensures the full compatibility of the plot
    colors and the colorbar. It's working principle is best understood
    with a few examples, available as a notebook in the examples directory.
    """

    def __init__(self, data=None, levels=None, nlevels=None, vmin=None,
                 vmax=None, extend=None, cmap=None):
        """Instanciate.

        Parameters
        ----------
        see the set_* functions
        """
        self.set_data(data)
        self.set_levels(levels)
        self.set_nlevels(nlevels)
        self.set_vmin(vmin)
        self.set_vmax(vmax)
        self.set_extend(extend)
        self.set_cmap(cmap)

    def update(self, d):
        """
        Update the properties of :class:`DataLevels` from the dictionary *d*.
        """

        for k, v in six.iteritems(d):
            func = getattr(self, 'set_' + k, None)
            if func is None or not six.callable(func):
                raise AttributeError('Unknown property %s' % k)
            func(v)

    def set_data(self, data=None):
        """Any kind of data array (also masked)."""
        if data is not None:
            self.data = np.ma.masked_invalid(np.atleast_1d(data), copy=False)
        else:
            self.data = np.ma.asarray([0., 1.])

    def set_levels(self, levels=None):
        """Levels you define. Must be monotically increasing."""
        self._levels = levels

    def set_nlevels(self, nlevels=None):
        """Automatic N levels. Ignored if set_levels has been set."""
        self._nlevels = nlevels

    def set_vmin(self, val=None):
        """Mininum level value. Ignored if set_levels has been set."""
        self._vmin = val

    def set_vmax(self, val=None):
        """Maximum level value. Ignored if set_levels has been set."""
        self._vmax = val

    def set_cmap(self, cm=None):
        """Set a colormap."""
        if cm is not None:
            self.cmap = cm
        else:
            self.cmap = mpl.colors.ListedColormap(['white'])

    def set_extend(self, extend=None):
        """Colorbar extensions: 'neither' | 'both' | 'min' | 'max'"""
        self._extend = extend

    def set_plot_params(self, levels=None, nlevels=None, vmin=None, vmax=None,
                        extend=None):
        """Shortcut to all parameters related to the plot.

        As a side effect, running set_plot_params() without arguments will
        reset the default behavior
        """
        self.set_vmin(vmin)
        self.set_vmax(vmax)
        self.set_levels(levels)
        self.set_nlevels(nlevels)
        self.set_extend(extend)

    @property
    def levels(self):
        """Clever getter."""
        levels = self._levels
        nlevels = self._nlevels
        if levels is not None:
            self.set_vmin(levels[0])
            self.set_vmax(levels[-1])
            return levels
        else:
            if nlevels is None:
                nlevels = 8
            if self.vmax == self.vmin:
                return np.linspace(self.vmin, self.vmax+1, nlevels)
            return np.linspace(self.vmin, self.vmax, nlevels)

    @property
    def nlevels(self):
        """Clever getter."""
        return len(self.levels)

    @property
    def vmin(self):
        """Clever getter."""
        if self._vmin is None:
            return np.min(self.data)
        else:
            return self._vmin

    @property
    def vmax(self):
        """Clever getter."""
        if self._vmax is None:
            return np.max(self.data)
        else:
            return self._vmax

    @property
    def extend(self):
        """Clever getter."""
        if self._extend is None:
            # If the user didnt set it, we decide
            maxd, mind = np.max(self.data), np.min(self.data)
            if maxd > self.vmax and mind < self.vmin:
                out = 'both'
            elif maxd > self.vmax:
                out = 'max'
            elif mind < self.vmin:
                out = 'min'
            else:
                out = 'neither'
            return out
        else:
            return self._extend

    @property
    def norm(self):
        """Clever getter."""
        l = self.levels
        e = self.extend
        # Warnings
        if e not in ['both', 'min'] and (np.min(l) > np.min(self.data)):
            warnings.warn('Minimum data out of bounds.', RuntimeWarning)
        if e not in ['both', 'max'] and (np.max(l) < np.max(self.data)):
            warnings.warn('Maximum data out of bounds.', RuntimeWarning)
        return ExtendedNorm(l, self.cmap.N, extend=e)

    def to_rgb(self):
        """Transform the data to RGB triples."""
        return self.cmap(self.norm(self.data))

    def colorbarbase(self, cax, **kwargs):
        """Returns a ColorbarBase to add to the cax axis. All keywords are
        passed to matplotlib.colorbar.ColorbarBase
        """

        # This is a discutable choice: with more than 60 colors (could be
        # less), we assume a continuous colorbar.
        if self.nlevels < 60:
            norm = self.norm
        else:
            norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        return mpl.colorbar.ColorbarBase(cax, extend=self.extend,
                                         cmap=self.cmap, norm=norm, **kwargs)

    def append_colorbar(self, ax, position='right', size='5%', pad=0.5):
        """Shortcut to append a colorbar to existing axes using matplotlib's
        make_axes_locatable toolkit.

        Parameters
        ----------
        ax: the axis to append the colorbar to
        position: "left"|"right"|"bottom"|"top"
        size: the size of the colorbar (e.g. in % of the ax)
        pad: pad between axes given in inches or tuple-like of floats,
             (horizontal padding, vertical padding)
        """

        orientation = 'horizontal'
        if position in ['left', 'right']:
            orientation = 'vertical'
        cax = make_axes_locatable(ax).append_axes(position, size=size, pad=pad)
        return self.colorbarbase(cax, orientation=orientation)

    def plot(self, ax):
        """Add a kind of plot of the data to an axis.

        More useful for child classes if you ask me but still.
        """
        data = np.atleast_2d(self.data)
        toplot = self.cmap(self.norm(data))
        ax.imshow(toplot, interpolation='none', origin='lower')

    def visualize(self, ax=None, title=None, orientation='vertical',
                  add_values=False, addcbar=True):
        """Quick'n dirty plot of the datalevels. Useful for debugging.

        Parameters
        ----------
        ax: the axis to add the plot to (optinal)
        title: the plot title
        orientation: the colorbar's orientation
        add_values: add the data values as text in the pixels (for testing)
        """

        # Do we make our own fig?
        _do_tight_layout = False
        if ax is None:
            fig, ax = plt.subplots(1)
            _do_tight_layout = True

        # Plot
        self.plot(ax)

        # Colorbar
        addcbar = self.vmin != self.vmax
        if addcbar:
            if orientation == 'horizontal':
                self.append_colorbar(ax, "top", size=0.2, pad=0.5)
            else:
                self.append_colorbar(ax, "right", size="5%", pad=0.2)

        # Mini add-on
        if add_values:
            data = np.atleast_2d(self.data)
            x, y = np.meshgrid(np.arange(data.shape[1]),
                               np.arange(data.shape[0]))
            for v, i, j in zip(data.flatten(), x.flatten(), y.flatten()):
                ax.text(i, j, v, horizontalalignment='center',
                        verticalalignment='center')

        # Details
        if title is not None:
            ax.set_title(title)
        if _do_tight_layout:
            fig.tight_layout()


class Map(DataLevels):
    """Plotting georeferenced data.

    A Map is an implementation of DataLevels that wraps imshow(), by adding
    all kinds of geoinformation on the plot. It's primary purpose is to add
    country borders or topographical shading. Another purpose is
    to be able to plot all kind of georeferenced data on an existing map
    while being sure that the plot is accurate.

    In short: cleo.Map is a higher-level, less wordy and less flexible
    version of cartopy or basemap. It's usefulness is best shown by the
    notebooks in the `examples` directory.

    For worldwide maps you'd better use the two libs above, because a cleo.Map
    is sensibly constrained by it's "squareness".
    """

    def __init__(self, grid, nx=500, ny=None, countries=True, **kwargs):
        """Make a new map.

        Parameters
        ----------
        grid: a salem.Grid instance defining the map
        nx: x resolution (in pixels) of the map
        ny: y resolution (in pixels) of the map (ignored if nx is set)
        countries: automatically add country borders to the map (you can do
        it later with a call to set_shapefile)
        kwards: all keywords accepted by DataLevels
        """

        self.grid = grid.center_grid.regrid(nx=nx, ny=ny)
        self.origin = 'lower' if self.grid.order == 'll' else 'upper'

        DataLevels.__init__(self, **kwargs)

        self._collections = []
        self._geometries = []
        self._text = []
        self.set_shapefile(countries=countries)
        self.set_lonlat_contours()
        self._shading_base()
        self._rgb = None
        self._contourf_data = None

    def _check_data(self, data=None, crs=None, interp='nearest',
                    overplot=False):
        """Interpolates the data to the map grid."""

        data = np.ma.fix_invalid(np.squeeze(data))
        shp = data.shape
        if len(shp) != 2:
            raise ValueError('Data should be 2D.')

        crs = gis.check_crs(crs)
        if crs is None:
            # Reform case, but with a sanity check
            if not np.isclose(shp[0] / shp[1], self.grid.ny / self.grid.nx,
                              atol=1e-2):
                raise ValueError('Dimensions of data do not match the map.')

            # need to resize if not same
            if not ((shp[0] == self.grid.ny) and (shp[1] == self.grid.nx)):
                if interp.lower() == 'linear':
                    interp = 'bilinear'
                if interp.lower() == 'spline':
                    interp = 'cubic'
                # TODO: this does not work well with masked arrays
                data = imresize(data.filled(np.NaN),
                                (self.grid.ny, self.grid.nx),
                                interp=interp, mode='F')
        elif isinstance(crs, Grid):
            # Remap
            if overplot:
                data = self.grid.map_gridded_data(data, crs, interp=interp,
                                                  out=self.data)
            else:
                data = self.grid.map_gridded_data(data, crs, interp=interp)
        else:
            raise ValueError('crs not understood')
        return data

    def set_data(self, data=None, crs=None, interp='nearest',
                 overplot=False):
        """Adds data to the plot. The data has to be georeferenced, i.e. by
        setting crs (if omitted the data is assumed to be defined on the
        map's grid)

        Parameters
        ----------
        data: the data array (2d)
        crs: the data coordinate reference system
        interp: 'nearest' (default) or 'linear', the interpolation algorithm
        overplot: add the data to an existing plot (useful for mosaics for
        example)
        """

        # Check input
        if data is None:
            self.data = np.ma.zeros((self.grid.ny, self.grid.nx))
            return
        data = self._check_data(data=data, crs=crs, interp=interp,
                                overplot=overplot)
        DataLevels.set_data(self, data)

    def set_contourf(self, data=None, crs=None, interp='nearest', **kwargs):
        """Adds data to contour on the map.

        Parameters
        ----------
        mask: bool array (2d)
        crs: the data coordinate reference system
        interp: 'nearest' (default) or 'linear', the interpolation algorithm
        kwargs: anything accepted by contourf
        """


        # Check input
        if data is None:
            self._contourf_data = None
            return

        self._contourf_data = self._check_data(data=data, crs=crs,
                                               interp=interp)
        self._contourf_kw = kwargs

    def set_geometry(self, geometry=None, crs=wgs84, text=None,
                     text_delta=(0.01, 0.01), text_kwargs=dict(), **kwargs):
        """Adds any Shapely geometry to the map (including polygons,
        points, etc.) If called without arguments, it removes all previous
        geometries.

        Parameters
        ----------
        geometry: a Shapely gometry object (must be a scalar!)
        crs: the associated coordinate reference system (default wgs84)
        text: if you want to add a text to the geometry (it's position is
        based on the geometry's centroid)
        text_delta: it can be useful to shift the text of a certain amount
        when annotating points. units are percentage of data coordinates.
        text_kwargs: the keyword arguments to pass to the test() function
        kwargs: any keyword associated with the geometrie's plotting function::
            - Point: all keywords accepted by scatter(): marker, s, edgecolor,
             facecolor...
            - Line: all keywords accepted by plot(): color, linewidth...
            - Polygon: all keywords accepted by PathPatch(): color, edgecolor,
             facecolor, linestyle, linewidth, alpha...
        """

        # Reset?
        if geometry is None:
            self._geometries = []
            return

        # Transform
        geom = gis.transform_geometry(geometry, crs=crs,
                                      to_crs=self.grid.center_grid)

        # Text
        if text is not None:
            x, y = geom.centroid.xy
            x = x[0] + text_delta[0] * self.grid.nx
            sign = self.grid.dy / np.abs(self.grid.dy)
            y = y[0] + text_delta[1] * self.grid.ny * sign
            self.set_text(x, y, text, crs=self.grid.center_grid,
                          **text_kwargs)

        # Save
        if 'Multi' in geom.type:
            for g in geom:
                self._geometries.append((g, kwargs))
                # dirty solution: I should use collections instead
                if 'label' in kwargs:
                    kwargs = kwargs.copy()
                    del kwargs['label']
        else:
            self._geometries.append((geom, kwargs))

    def set_points(self, x, y, **kwargs):
        """Shortcut for set_geometry() accepting coordinates as input."""
        self.set_geometry(MultiPoint(np.array([x, y]).T), **kwargs)

    def set_text(self, x=None, y=None, text='', crs=wgs84, **kwargs):
        """Add a text to the map.

        Keyword arguments will be passed to mpl's text() function.
        """

        # Reset?
        if x is None:
            self._text = []
            return

        # Transform
        x, y = self.grid.center_grid.transform(x, y, crs=crs)
        self._text.append((x, y, text, kwargs))

    def set_shapefile(self, shape=None, countries=False, oceans=False,
                      rivers=False, **kwargs):
        """Add a shapefile to the plot.

        Cleo is shipped with a few default settings for country borders,
        oceans and rivers (set one at the time!)

        set_shapefile() without argument will reset the map to zero shapefiles.

        Parameters
        ----------
        shape: the path to the shapefile to read
        countries: if True, add country borders
        oceans: if True, add oceans
        rivers: if True, add rivers
        kwargs: all keywords accepted by the corresponding collection.
        For LineStrings::
            linewidths, colors, linestyles, ...
        For Polygons::
            alpha, edgecolor, facecolor, fill, linestyle, linewidth, color, ...
        """

        # See if the user wanted defaults settings
        if oceans:
            kwargs.setdefault('facecolor', (0.36862745, 0.64313725, 0.8))
            kwargs.setdefault('edgecolor', 'none')
            kwargs.setdefault('alpha', 1)
            return self.set_shapefile(shapefiles['oceans'], **kwargs)
        if rivers:
            kwargs.setdefault('colors', (0.08984375, 0.65625, 0.8515625))
            return self.set_shapefile(shapefiles['rivers'], **kwargs)
        if countries:
            return self.set_shapefile(shapefiles['world_borders'])

        # Reset?
        if shape is None:
            self._collections = []
            return

        # Transform
        shape = sio.read_shapefile_to_grid(shape, grid=self.grid)
        if len(shape) == 0:
            return

        # Different collection for each type
        geomtype = shape.iloc[0].geometry.type
        if 'Polygon' in geomtype:
            patches = []
            for g in shape.geometry:
                if 'Multi' in g.type:
                    for gg in g:
                        patches.append(PolygonPatch(gg))
                else:
                    patches.append(PolygonPatch(g))
            kwargs.setdefault('facecolor', 'none')
            self._collections.append(PatchCollection(patches, **kwargs))
        elif 'LineString' in geomtype:
            lines = []
            for g in shape.geometry:
                if 'Multi' in g.type:
                    for gg in g:
                        lines.append(np.array(gg))
                else:
                    lines.append(np.array(g))
            self._collections.append(LineCollection(lines, **kwargs))
        else:
            raise NotImplementedError(geomtype)

    def _find_interval(self):
        """Quick n dirty function to find a suitable lonlat interval."""
        candidates = [0.001, 0.002, 0.005,
                      0.01, 0.02, 0.05,
                      0.1, 0.2, 0.5,
                      1, 2, 5, 10, 20]
        xx, yy = self.grid.pixcorner_ll_coordinates
        for inter in candidates:
            _xx = xx / inter
            _yy = yy / inter
            mm_x = [np.ceil(np.min(_xx)), np.floor(np.max(_xx))]
            mm_y = [np.ceil(np.min(_yy)), np.floor(np.max(_yy))]
            nx = mm_x[1]-mm_x[0]+1
            ny = mm_y[1]-mm_y[0]+1
            if np.max([nx, ny]) <= 8:
                break
        return inter

    def set_lonlat_contours(self, interval=None, xinterval=None,
                            yinterval=None, add_tick_labels=True,
                            **kwargs):
        """Add longitude and latitude contours to the map.

        Parameters
        ----------
        interval: interval (in degrees) between the contours (same for lon
        and lat)
        xinterval: set a different interval for lons
        yinterval: set a different interval for lats
        add_tick_label: add the ticks labels to the map
        kwargs: any keyword accepted by contour()
        """

        # Defaults
        if interval is None:
            interval = self._find_interval()
        if xinterval is None:
            xinterval = interval
        if yinterval is None:
            yinterval = interval

        # Change XY into interval coordinates, and back after rounding
        xx, yy = self.grid.pixcorner_ll_coordinates
        _xx = xx / xinterval
        _yy = yy / yinterval
        mm_x = [np.ceil(np.min(_xx)), np.floor(np.max(_xx))]
        mm_y = [np.ceil(np.min(_yy)), np.floor(np.max(_yy))]
        self.xtick_levs = (mm_x[0] + np.arange(mm_x[1]-mm_x[0]+1)) * xinterval
        self.ytick_levs = (mm_y[0] + np.arange(mm_y[1]-mm_y[0]+1)) * yinterval

        # Decide on float format
        d = np.array(['4', '3', '2', '1', '0'])
        d = d[interval < np.array([0.001, 0.01, 0.1, 1, 10000])][0]

        # The labels (quite ugly)
        self.xtick_pos = []
        self.xtick_val = []
        self.ytick_pos = []
        self.ytick_val = []
        if add_tick_labels:
            _xx = xx[0 if self.origin == 'lower' else -1, :]
            _xi = np.arange(self.grid.nx+1)
            for xl in self.xtick_levs:
                if (xl > _xx[-1]) or (xl < _xx[0]):
                    continue
                self.xtick_pos.append(np.interp(xl, _xx, _xi))
                label = ('{:.' + d + 'f}').format(xl)
                label += 'W' if (xl < 0) else 'E'
                if xl == 0:
                    label = '0'
                self.xtick_val.append(label)

            _yy = np.sort(yy[:, 0])
            _yi = np.arange(self.grid.ny+1)
            if self.origin == 'upper':
                _yi = _yi[::-1]
            for yl in self.ytick_levs:
                if (yl > _yy[-1]) or (yl < _yy[0]):
                    continue
                self.ytick_pos.append(np.interp(yl, _yy, _yi))
                label = ('{:.' + d + 'f}').format(yl)
                label += 'S' if (yl < 0) else 'N'
                if yl == 0:
                    label = 'Eq.'
                self.ytick_val.append(label)

        # Done
        kwargs.setdefault('colors', 'gray')
        kwargs.setdefault('linestyles', 'dashed')
        self.ll_contour_kw = kwargs

    def _shading_base(self, slope=None, relief_factor=0.7):
        """Compute the shading factor out of the slope."""

        # reset?
        if slope is None:
            self.slope = None
            return

        # I got this formula from D. Scherer. It works and I dont know why
        p = np.nonzero(slope > 0)
        if len(p[0]) > 0:
            temp = np.clip(slope[p] / (2 * np.std(slope)), -1, 1)
            slope[p] = 0.4 * np.sin(0.5*np.pi*temp)
        self.relief_factor = relief_factor
        self.slope = slope

    def set_topography(self, topo=None, crs=None, relief_factor=0.7, **kwargs):
        """Add topographical shading to the map.

        Parameters
        ----------
        topo: path to a geotiff file containing the topography, OR
              2d data array
        relief_factor: how strong should the shading be?
        kwargs: any keyword accepted by salem.Grid.map_gridded_data (interp,ks)

        Returns
        -------
        the topography if needed (bonus)
        """

        if topo is None:
            self._shading_base()
        kwargs.setdefault('interp', 'spline')

        if isinstance(topo, six.string_types):
            _, ext = os.path.splitext(topo)
            if ext.lower() == '.tif':
                g = GeoTiff(topo)
                # Spare memory
                ex = self.grid.extent_in_crs(crs=wgs84)  # l, r, b, t
                g.set_subset(corners=((ex[0], ex[2]), (ex[1], ex[3])),
                             crs=wgs84, margin=10)
                z = g.get_vardata()
                z[z < -999] = 0
                z = self.grid.map_gridded_data(z, g.grid, **kwargs)
            else:
                raise ValueError('File extension not recognised: {}'
                                 .format(ext))
        else:
            z = self._check_data(topo, crs=crs, **kwargs)

        # Gradient in m m-1
        ddx = self.grid.dx
        ddy = self.grid.dy
        if self.grid.proj.is_latlong():
            # we make a coarse approx of the avg dx on a sphere
            _, lat = self.grid.ll_coordinates
            ddx = np.mean(ddx * 111200 * np.cos(lat * np.pi / 180))
            ddy *= 111200

        dy, dx = np.gradient(z, ddy, ddx)
        self._shading_base(dx - dy, relief_factor=relief_factor)
        return z

    def set_rgb(self, img=None, crs=None):
        """Manually force to a rgb img"""

        if (len(img.shape) != 3) or (img.shape[-1] != 3):
            raise ValueError('img should be of shape (x, y, 3)')

        # Unefficient but by far easiest right now
        out = []
        for i in [0, 1, 2]:
            out.append(self._check_data(img[..., i], crs=crs))
        self._rgb = np.dstack(out)

    def to_rgb(self):
        """Transform the data to a RGB image and add topographical shading."""

        if self._rgb is None:
            toplot = DataLevels.to_rgb(self)
        else:
            toplot = self._rgb

        # Shading
        if self.slope is not None:
            # remove alphas?
            try:
                pno = np.where(toplot[:, :, 3] == 0.)
                for i in [0, 1, 2]:
                    toplot[pno[0], pno[1], i] = 1
                toplot[:, :, 3] = 1
            except IndexError:
                pass

            # Actual shading
            level = 1.0 - 0.1 * self.relief_factor
            sens = 1 + 0.7 * self.relief_factor * self.slope
            for i in [0, 1, 2]:
                toplot[:, :, i] = np.clip(level * toplot[:, :, i] * sens, 0, 1)

        # OK!
        return toplot

    def plot(self, ax):
        """Add the map plot to an axis.

        It first plots the image and then adds all the cartographic
        information on top of it.
        """

        # Image is the easiest
        ax.imshow(self.to_rgb(), interpolation='none', origin=self.origin)
        ax.autoscale(False)

        # Stippling
        if self._contourf_data is not None:
            ax.contourf(self._contourf_data, **self._contourf_kw)

        # Shapefiles
        for col in self._collections:
            ax.add_collection(copy.copy(col))

        # Lon lat contours
        lon, lat = self.grid.pixcorner_ll_coordinates
        if len(self.xtick_levs) > 0:
            ax.contour(lon, levels=self.xtick_levs,
                       extent=(-0.5, self.grid.nx-0.5, -0.5, self.grid.ny),
                       **self.ll_contour_kw)
        if len(self.ytick_levs) > 0:
            ax.contour(lat, levels=self.ytick_levs,
                       extent=(-0.5, self.grid.nx, -0.5, self.grid.ny-0.5),
                       **self.ll_contour_kw)

        # Geometries
        for g, kwargs in self._geometries:
            if g.type == 'Polygon':
                kwargs.setdefault('facecolor', 'none')
                plot_polygon(ax, g, **kwargs)  # was g.buffer(0). Why?
            if g.type in ['LineString', 'LinearRing']:
                a = np.array(g)
                kwargs.setdefault('color', 'k')
                ax.plot(a[:, 0], a[:, 1], **kwargs)
            if g.type == 'Point':
                kwargs.setdefault('marker', 'o')
                kwargs.setdefault('s', 60)
                kwargs.setdefault('facecolor', 'w')
                kwargs.setdefault('edgecolor', 'k')
                kwargs.setdefault('linewidths', 1)
                if 'markersize' in kwargs:
                    # For those tempted to use the whole kw
                    kwargs['s'] = kwargs['markersize']
                    del kwargs['markersize']
                if 'color' in kwargs:
                    # For those tempted to use the whole kw
                    kwargs['facecolor'] = kwargs['color']
                    kwargs['edgecolor'] = kwargs['color']
                if 'c' in kwargs:
                    # For those tempted to use the whole kw
                    kwargs['facecolor'] = kwargs['c']
                    kwargs['edgecolor'] = kwargs['c']
                    del kwargs['c']
                ax.scatter(g.x, g.y, **kwargs)

        # Texts
        for x, y, s, kwargs in self._text:
            ax.text(x, y, s, **kwargs)

        # Ticks
        if (len(self.xtick_pos) > 0) or (len(self.ytick_pos) > 0):
            ax.xaxis.set_ticks(np.array(self.xtick_pos)-0.5)
            ax.yaxis.set_ticks(np.array(self.ytick_pos)-0.5)
            ax.set_xticklabels(self.xtick_val)
            ax.set_yticklabels(self.ytick_val)
        else:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])


def plot_polygon(ax, poly, edgecolor='black', **kwargs):
    """ Plot a single Polygon geometry """

    a = np.asarray(poly.exterior)
    # without Descartes, we could make a Patch of exterior
    ax.add_patch(PolygonPatch(poly, **kwargs))
    ax.plot(a[:, 0], a[:, 1], color=edgecolor)
    for p in poly.interiors:
        x, y = zip(*p.coords)
        ax.plot(x, y, color=edgecolor)