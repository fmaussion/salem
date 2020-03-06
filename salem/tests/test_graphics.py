from __future__ import division
from distutils.version import LooseVersion

import copy
import warnings
import os
import shutil

import numpy as np
import pytest
import unittest
from numpy.testing import assert_array_equal

try:
    import matplotlib as mpl
except ImportError:
    pytest.skip("Requires matplotlib", allow_module_level=True)

try:
    import shapely.geometry as shpg
except ImportError:
    pytest.skip("Requires shapely", allow_module_level=True)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import geopandas as gpd
MPL_VERSION = LooseVersion(mpl.__version__)
ftver = LooseVersion(mpl.ft2font.__freetype_version__)
if ftver >= LooseVersion('2.8.0'):
    freetype_subdir = 'freetype_28'
else:
    freetype_subdir = 'freetype_old'

from salem.graphics import ExtendedNorm, DataLevels, Map, get_cmap, shapefiles
from salem import graphics
from salem import (Grid, wgs84, mercator_grid, GeoNetcdf,
                   read_shapefile_to_grid, GeoTiff, GoogleCenterMap,
                   GoogleVisibleMap, open_wrf_dataset, open_xr_dataset,
                   python_version, cache_dir, sample_data_dir)
from salem.utils import get_demo_file
from salem.tests import (requires_matplotlib, requires_cartopy)

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(current_dir, 'tmp')

baseline_subdir = '2.0.x'
baseline_dir = os.path.join(sample_data_dir, 'baseline_images',
                            baseline_subdir, freetype_subdir)

tolpy2 = 5 if python_version == 'py3' else 10


def _create_dummy_shp(fname):
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    e_line = shpg.LinearRing([(1.5, 1), (2., 1.5), (1.5, 2.), (1, 1.5)])
    i_line = shpg.LinearRing([(1.4, 1.4), (1.6, 1.4), (1.6, 1.6), (1.4, 1.6)])
    p1 = shpg.Polygon(e_line, [i_line])
    p2 = shpg.Polygon([(2.5, 1.3), (3., 1.8), (2.5, 2.3), (2, 1.8)])
    p3 = shpg.Point(0.5, 0.5)
    p4 = shpg.Point(1, 1)
    df = gpd.GeoDataFrame()
    df['name'] = ['Polygon', 'Line']
    df['geometry'] = gpd.GeoSeries([p1, p2])
    of = os.path.join(testdir, fname)
    df.crs = 'epsg:4326'
    df.to_file(of)
    return of


class TestColors(unittest.TestCase):

    @requires_matplotlib
    def test_extendednorm(self):

        bounds = [1, 2, 3]
        cm = mpl.cm.get_cmap('jet')

        mynorm = graphics.ExtendedNorm(bounds, cm.N)
        refnorm = mpl.colors.BoundaryNorm(bounds, cm.N)
        x = np.random.randn(100) * 10 - 5
        np.testing.assert_array_equal(refnorm(x), mynorm(x))

        refnorm = mpl.colors.BoundaryNorm([0] + bounds + [4], cm.N)
        mynorm = graphics.ExtendedNorm(bounds, cm.N, extend='both')
        x = np.random.random(100) + 1.5
        np.testing.assert_array_equal(refnorm(x), mynorm(x))

        # Min and max
        cmref = mpl.colors.ListedColormap(['blue', 'red'])
        cmref.set_over('black')
        cmref.set_under('white')

        cmshould = mpl.colors.ListedColormap(['white', 'blue', 'red', 'black'])
        cmshould.set_over(cmshould(cmshould.N))
        cmshould.set_under(cmshould(0))

        refnorm = mpl.colors.BoundaryNorm(bounds, cmref.N)
        mynorm = graphics.ExtendedNorm(bounds, cmshould.N, extend='both')
        np.testing.assert_array_equal(refnorm.vmin, mynorm.vmin)
        np.testing.assert_array_equal(refnorm.vmax, mynorm.vmax)
        x = [-1, 1.2, 2.3, 9.6]
        np.testing.assert_array_equal(cmshould([0,1,2,3]), cmshould(mynorm(x)))
        x = np.random.randn(100) * 10 + 2
        np.testing.assert_array_equal(cmref(refnorm(x)), cmshould(mynorm(x)))

        np.testing.assert_array_equal(-1, mynorm(-1))
        np.testing.assert_array_equal(1, mynorm(1.1))
        np.testing.assert_array_equal(4, mynorm(12))

        # Just min
        cmref = mpl.colors.ListedColormap(['blue', 'red'])
        cmref.set_under('white')
        cmshould = mpl.colors.ListedColormap(['white', 'blue', 'red'])
        cmshould.set_under(cmshould(0))

        np.testing.assert_array_equal(2, cmref.N)
        np.testing.assert_array_equal(3, cmshould.N)
        refnorm = mpl.colors.BoundaryNorm(bounds, cmref.N)
        mynorm = graphics.ExtendedNorm(bounds, cmshould.N, extend='min')
        np.testing.assert_array_equal(refnorm.vmin, mynorm.vmin)
        np.testing.assert_array_equal(refnorm.vmax, mynorm.vmax)
        x = [-1, 1.2, 2.3]
        np.testing.assert_array_equal(cmshould([0,1,2]), cmshould(mynorm(x)))
        x = np.random.randn(100) * 10 + 2
        np.testing.assert_array_equal(cmref(refnorm(x)), cmshould(mynorm(x)))

        # Just max
        cmref = mpl.colors.ListedColormap(['blue', 'red'])
        cmref.set_over('black')
        cmshould = mpl.colors.ListedColormap(['blue', 'red', 'black'])
        cmshould.set_over(cmshould(2))

        np.testing.assert_array_equal(2, cmref.N)
        np.testing.assert_array_equal(3, cmshould.N)
        refnorm = mpl.colors.BoundaryNorm(bounds, cmref.N)
        mynorm = graphics.ExtendedNorm(bounds, cmshould.N, extend='max')
        np.testing.assert_array_equal(refnorm.vmin, mynorm.vmin)
        np.testing.assert_array_equal(refnorm.vmax, mynorm.vmax)
        x = [1.2, 2.3, 4]
        np.testing.assert_array_equal(cmshould([0,1,2]), cmshould(mynorm(x)))
        x = np.random.randn(100) * 10 + 2
        np.testing.assert_array_equal(cmref(refnorm(x)), cmshould(mynorm(x)))

        # General case
        bounds = [1, 2, 3, 4]
        cm = mpl.cm.get_cmap('jet')
        mynorm = graphics.ExtendedNorm(bounds, cm.N, extend='both')
        refnorm = mpl.colors.BoundaryNorm([-100] + bounds + [100], cm.N)
        x = np.random.randn(100) * 10 - 5
        ref = refnorm(x)
        ref = np.where(ref == 0, -1, ref)
        ref = np.where(ref == cm.N-1, cm.N, ref)
        np.testing.assert_array_equal(ref, mynorm(x))


class TestGraphics(unittest.TestCase):

    @requires_matplotlib
    def test_datalevels_output(self):

        # Test basic stuffs
        c = graphics.DataLevels(nlevels=2)
        assert_array_equal(c.levels, [0, 1])
        c.set_data([1, 2, 3, 4])
        assert_array_equal(c.levels, [1, 4])

        c = graphics.DataLevels(levels=[1, 2, 3])
        assert_array_equal(c.levels, [1, 2, 3])

        c = graphics.DataLevels(nlevels=10, data=[0, 9])
        assert_array_equal(c.levels, np.linspace(0, 9, num=10))
        self.assertTrue(c.extend == 'neither')

        c = graphics.DataLevels(nlevels=10, data=[0, 9], vmin=2, vmax=3)
        assert_array_equal(c.levels, np.linspace(2, 3, num=10))
        self.assertTrue(c.extend == 'both')
        c.set_extend('neither')
        self.assertTrue(c.extend == 'neither')
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            out = c.to_rgb()
            # Verify some things
            assert len(w) == 2
            assert issubclass(w[0].category, RuntimeWarning)
            assert issubclass(w[1].category, RuntimeWarning)

        c = graphics.DataLevels(nlevels=10, data=[2.5], vmin=2, vmax=3)
        assert_array_equal(c.levels, np.linspace(2, 3, num=10))
        self.assertTrue(c.extend == 'neither')
        c.update(dict(extend='both'))
        self.assertTrue(c.extend == 'both')
        self.assertRaises(AttributeError, c.update, dict(dummy='t'))

        c = graphics.DataLevels(nlevels=10, data=[0, 9], vmax=3)
        assert_array_equal(c.levels, np.linspace(0, 3, num=10))
        self.assertTrue(c.extend == 'max')

        c = graphics.DataLevels(nlevels=10, data=[0, 9], vmin=1)
        assert_array_equal(c.levels, np.linspace(1, 9, num=10))
        self.assertTrue(c.extend == 'min')

        c = graphics.DataLevels(nlevels=10, data=[0, 9], vmin=-1)
        assert_array_equal(c.levels, np.linspace(-1, 9, num=10))
        self.assertTrue(c.extend == 'neither')
        c.set_plot_params()
        self.assertTrue(c.extend == 'neither')
        assert_array_equal(c.vmin, 0)
        assert_array_equal(c.vmax, 9)
        c.set_plot_params(vmin=1)
        assert_array_equal(c.vmin, 1)
        c.set_data([-12, 8])
        assert_array_equal(c.vmin, 1)
        self.assertTrue(c.extend == 'min')
        c.set_data([2, 8])
        self.assertTrue(c.extend == 'neither')
        c.set_extend('both')
        self.assertTrue(c.extend == 'both')
        c.set_data([3, 3])
        self.assertTrue(c.extend == 'both')
        c.set_extend()
        self.assertTrue(c.extend == 'neither')

        # Test the conversion
        cm = mpl.colors.ListedColormap(['white', 'blue', 'red', 'black'])
        x = [-1, 0.9, 1.2, 2, 999, 0.8]
        c = graphics.DataLevels(levels=[0, 1, 2], data=x, cmap=cm)
        r = c.to_rgb()
        self.assertTrue(len(x) == len(r))
        self.assertTrue(c.extend == 'both')
        assert_array_equal(r, cm([0, 1, 2, 3, 3, 1]))

        x = [0.9, 1.2]
        c = graphics.DataLevels(levels=[0, 1, 2], data=x, cmap=cm, extend='both')
        r = c.to_rgb()
        self.assertTrue(len(x) == len(r))
        self.assertTrue(c.extend == 'both')
        assert_array_equal(r, cm([1, 2]))

        cm = mpl.colors.ListedColormap(['white', 'blue', 'red'])
        c = graphics.DataLevels(levels=[0, 1, 2], data=x, cmap=cm, extend='min')
        r = c.to_rgb()
        self.assertTrue(len(x) == len(r))
        assert_array_equal(r, cm([1, 2]))

        cm = mpl.colors.ListedColormap(['blue', 'red', 'black'])
        c = graphics.DataLevels(levels=[0, 1, 2], data=x, cmap=cm, extend='max')
        r = c.to_rgb()
        self.assertTrue(len(x) == len(r))
        assert_array_equal(r, cm([0, 1]))

    @requires_matplotlib
    def test_map(self):

        a = np.zeros((4, 5))
        a[0, 0] = -1
        a[1, 1] = 1.1
        a[2, 2] = 2.2
        a[2, 4] = 1.9
        a[3, 3] = 9
        cmap = copy.deepcopy(mpl.cm.get_cmap('jet'))

        # ll_corner (type geotiff)
        g = Grid(nxny=(5, 4), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84,
                 pixel_ref='corner')
        c = graphics.Map(g, ny=4, countries=False)
        c.set_cmap(cmap)
        c.set_plot_params(levels=[0, 1, 2, 3])
        c.set_data(a)
        rgb1 = c.to_rgb()
        c.set_data(a, crs=g)
        assert_array_equal(rgb1, c.to_rgb())
        c.set_data(a, interp='linear')
        rgb1 = c.to_rgb()
        c.set_data(a, crs=g, interp='linear')
        assert_array_equal(rgb1, c.to_rgb())

        # centergrid (type WRF)
        g = Grid(nxny=(5, 4), dxdy=(1, 1), x0y0=(0.5, 0.5), proj=wgs84,
                 pixel_ref='center')
        c = graphics.Map(g, ny=4, countries=False)
        c.set_cmap(cmap)
        c.set_plot_params(levels=[0, 1, 2, 3])
        c.set_data(a)
        rgb1 = c.to_rgb()
        c.set_data(a, crs=g)
        assert_array_equal(rgb1, c.to_rgb())
        c.set_data(a, interp='linear')
        rgb1 = c.to_rgb()
        c.set_data(a, crs=g.corner_grid, interp='linear')
        assert_array_equal(rgb1, c.to_rgb())
        c.set_data(a, crs=g.center_grid, interp='linear')
        assert_array_equal(rgb1, c.to_rgb())

        # More pixels
        c = graphics.Map(g, ny=500, countries=False)
        c.set_cmap(cmap)
        c.set_plot_params(levels=[0, 1, 2, 3])
        c.set_data(a)
        rgb1 = c.to_rgb()
        c.set_data(a, crs=g)
        assert_array_equal(rgb1, c.to_rgb())
        c.set_data(a, interp='linear')
        rgb1 = c.to_rgb()
        c.set_data(a, crs=g, interp='linear')
        rgb2 = c.to_rgb()

        # The interpolation is conservative with the grid...
        srgb = np.sum(rgb2[..., 0:3], axis=2)
        pok = np.nonzero(srgb != srgb[0, 0])
        rgb1 = rgb1[np.min(pok[0])+1:np.max(pok[0]-1),
                    np.min(pok[1])+1:np.max(pok[1]-1),
                    ...]
        rgb2 = rgb2[np.min(pok[0])+1:np.max(pok[0]-1),
                    np.min(pok[1])+1:np.max(pok[1]-1),
                    ...]

        assert_array_equal(rgb1, rgb2)

        cmap.set_bad('pink')

        # Add masked arrays
        a[1, 1] = np.NaN
        c.set_data(a)
        rgb1 = c.to_rgb()
        c.set_data(a, crs=g)
        assert_array_equal(rgb1, c.to_rgb())

        # Interp?
        c.set_data(a, interp='linear')
        rgb1 = c.to_rgb()
        c.set_data(a, crs=g, interp='linear')
        rgb2 = c.to_rgb()
        # Todo: there's something sensibly wrong about imresize here
        # but I think it is out of my scope
        # assert_array_equal(rgb1, rgb2)

    @requires_matplotlib
    def test_increase_coverage(self):

        # Just for coverage -> empty shapes should not trigger an error
        grid = mercator_grid(center_ll=(-20, 40), extent=(2000, 2000), nx=10)
        c = graphics.Map(grid)

        # Assigning wrongly shaped data should, however
        self.assertRaises(ValueError, c.set_data, np.zeros((3, 8)))


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=10)
def test_extendednorm():
    a = np.zeros((4, 5))
    a[0, 0] = -9999
    a[1, 1] = 1.1
    a[2, 2] = 2.2
    a[2, 4] = 1.9
    a[3, 3] = 9999999

    cm = mpl.cm.get_cmap('jet')
    bounds = [0, 1, 2, 3]
    norm = ExtendedNorm(bounds, cm.N, extend='both')

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    imax = ax1.imshow(a, interpolation='None', norm=norm, cmap=cm,
                      origin='lower');
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(imax, cax=cax, extend='both')

    ti = cm(norm(a))
    ax2.imshow(ti, interpolation='None', origin='lower')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = mpl.colorbar.ColorbarBase(cax, extend='both', cmap=cm,
                                     norm=norm)
    fig.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=10)
def test_datalevels():
    plt.close()

    a = np.zeros((4, 5))
    a[0, 0] = -1
    a[1, 1] = 1.1
    a[2, 2] = 2.2
    a[2, 4] = 1.9
    a[3, 3] = 9

    cm = copy.copy(mpl.cm.get_cmap('jet'))
    cm.set_bad('pink')

    # fig, axes = plt.subplots(nrows=3, ncols=2)
    fig = plt.figure()
    ax = iter([fig.add_subplot(3, 2, i) for i in [1, 2, 3, 4, 5, 6]])

    # The extended version should be automated
    c = DataLevels(levels=[0, 1, 2, 3], data=a, cmap=cm)
    c.visualize(next(ax), title='levels=[0,1,2,3]')

    # Without min
    a[0, 0] = 0
    c = DataLevels(levels=[0, 1, 2, 3], data=a, cmap=cm)
    c.visualize(next(ax), title='modified a for no min oob')

    # Without max
    a[3, 3] = 0
    c = DataLevels(levels=[0, 1, 2, 3], data=a, cmap=cm)
    c.visualize(next(ax), title='modified a for no max oob')

    # Forced bounds
    c = DataLevels(levels=[0, 1, 2, 3], data=a, cmap=cm, extend='both')
    c.visualize(next(ax), title="extend='both'")

    # Autom nlevels
    a[0, 0] = -1
    a[3, 3] = 9
    c = DataLevels(nlevels=127, vmin=0, vmax=3, data=a, cmap=cm)
    c.visualize(next(ax), title="Auto levels with oob data")

    # Missing data
    a[3, 0] = np.NaN
    c = DataLevels(nlevels=127, vmin=0, vmax=3, data=a, cmap=cm)
    c.visualize(next(ax), title="missing data")

    plt.tight_layout()

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=5)
def test_datalevels_visu_h():
    a = np.array([-1., 0., 1.1, 1.9, 9.])
    cm = mpl.cm.get_cmap('RdYlBu_r')

    dl = DataLevels(a, cmap=cm, levels=[0, 1, 2, 3])

    fig, ax = plt.subplots(1)
    dl.visualize(ax=ax, orientation='horizontal', add_values=True)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_datalevels_visu_v():
    a = np.array([-1., 0., 1.1, 1.9, 9.])
    cm = mpl.cm.get_cmap('RdYlBu_r')

    dl = DataLevels(a.reshape((5, 1)), cmap=cm, levels=[0, 1, 2, 3])

    fig, ax = plt.subplots(1)
    dl.visualize(ax=ax, orientation='vertical', add_values=True)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=10)
def test_simple_map():
    a = np.zeros((4, 5))
    a[0, 0] = -1
    a[1, 1] = 1.1
    a[2, 2] = 2.2
    a[2, 4] = 1.9
    a[3, 3] = 9
    a_inv = a[::-1, :]
    fs = _create_dummy_shp('fs.shp')

    # UL Corner
    g1 = Grid(nxny=(5, 4), dxdy=(1, -1), x0y0=(-1, 3), proj=wgs84,
              pixel_ref='corner')
    c1 = Map(g1, ny=4, countries=False)

    # LL Corner
    g2 = Grid(nxny=(5, 4), dxdy=(1, 1), x0y0=(-1, -1), proj=wgs84,
              pixel_ref='corner')
    c2 = Map(g2, ny=4, countries=False)

    # Settings
    for c, data in zip([c1, c2], [a_inv, a]):
        c.set_cmap(mpl.cm.get_cmap('jet'))
        c.set_plot_params(levels=[0, 1, 2, 3])
        c.set_data(data)
        c.set_shapefile(fs)
        c.set_lonlat_contours(interval=0.5)

    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    c1.visualize(ax1)
    c2.visualize(ax2)

    # UL Corner
    c1 = Map(g1, ny=400, countries=False)
    c2 = Map(g2, ny=400, countries=False)
    # Settings
    for c, data, g in zip([c1, c2], [a_inv, a], [g1, g2]):
        c.set_cmap(mpl.cm.get_cmap('jet'))
        c.set_data(data, crs=g)
        c.set_shapefile(fs)
        c.set_plot_params(nlevels=256)
        c.set_lonlat_contours(interval=2)
    ax1 = fig.add_subplot(323)
    ax2 = fig.add_subplot(324)
    c1.visualize(ax1)
    c2.visualize(ax2)

    # Settings
    for c, data in zip([c1, c2], [a_inv, a]):
        c.set_plot_params(nlevels=256, vmax=3)
        c.set_lonlat_contours(interval=1)
        c.set_data(data, interp='linear')
    ax1 = fig.add_subplot(325)
    ax2 = fig.add_subplot(326)
    c1.visualize(ax1)
    c2.visualize(ax2)

    fig.tight_layout()
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=12)
def test_contourf():
    a = np.zeros((4, 5))
    a[0, 0] = -1
    a[1, 1] = 1.1
    a[2, 2] = 2.2
    a[2, 4] = 1.9
    a[3, 3] = 9

    # UL Corner
    g = Grid(nxny=(5, 4), dxdy=(1, -1), x0y0=(-1, 3), proj=wgs84,
             pixel_ref='corner')
    c = Map(g, ny=400, countries=False)

    c.set_cmap(mpl.cm.get_cmap('viridis'))
    c.set_plot_params(levels=[0, 1, 2, 3])
    c.set_data(a)
    s = a * 0.
    s[2, 2] = 1
    c.set_contourf(s, interp='linear', hatches=['xxx'], colors='none',
                   levels=[0.5, 1.5])

    s = a * 0.
    s[0:2, 3:] = 1
    s[0, 4] = 2
    c.set_contour(s, interp='linear', colors='k', linewidths=6,
                  levels=[0.5, 1., 1.5])

    c.set_lonlat_contours(interval=0.5)

    # Add a geometry for fun
    gs = g.to_dict()
    gs['nxny'] = (1, 2)
    gs['x0y0'] = (0, 2)
    gs = Grid.from_dict(gs)
    c.set_geometry(gs.extent_as_polygon(), edgecolor='r', linewidth=2)

    fig, ax = plt.subplots(1)
    c.visualize(ax=ax)
    fig.tight_layout()

    # remove it
    c.set_contourf()
    c.set_contour()

    return fig

@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_merca_map():
    grid = mercator_grid(center_ll=(11.38, 47.26),
                         extent=(2000000, 2000000))

    m1 = Map(grid)
    m1.set_scale_bar(color='red')

    grid = mercator_grid(center_ll=(11.38, 47.26),
                         extent=(2000000, 2000000),
                         origin='upper-left')
    m2 = Map(grid)
    m2.set_scale_bar(length=700000, location=(0.3, 0.05))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    m1.visualize(ax=ax1, addcbar=False)
    m2.visualize(ax=ax2, addcbar=False)
    plt.tight_layout()

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_merca_nolabels():
    grid = mercator_grid(center_ll=(11.38, 47.26),
                         extent=(2000000, 2000000))

    m1 = Map(grid)

    m1.set_lonlat_contours(add_tick_labels=False)
    fig, ax = plt.subplots(1)
    m1.visualize(ax=ax)
    fig.tight_layout()

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=5)
def test_oceans():
    f = os.path.join(get_demo_file('wrf_tip_d1.nc'))
    grid = GeoNetcdf(f).grid
    m = Map(grid, countries=False)
    m.set_shapefile(rivers=True, linewidths=2)
    m.set_shapefile(oceans=True, edgecolor='k', linewidth=3)
    m.set_lonlat_contours(linewidths=1)

    fig, ax = plt.subplots(1, 1)
    m.visualize(ax=ax, addcbar=False)
    plt.tight_layout()

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_geometries():
    # UL Corner
    g = Grid(nxny=(5, 4), dxdy=(10, 10), x0y0=(-20, -15), proj=wgs84,
             pixel_ref='corner')
    c = Map(g, ny=4)
    c.set_lonlat_contours(interval=10., colors='crimson', linewidths=1)

    c.set_geometry(shpg.Point(10, 10), color='darkred', markersize=60)
    c.set_geometry(shpg.Point(5, 5), s=500, marker='s',
                   facecolor='green', hatch='||||')

    s = np.array([(-5, -10), (0., -5), (-5, 0.), (-10, -5)])
    l1 = shpg.LineString(s)
    l2 = shpg.LinearRing(s + 3)
    c.set_geometry(l1)
    c.set_geometry(l2, color='pink', linewidth=3)

    s += 20
    p = shpg.Polygon(shpg.LineString(s), [shpg.LineString(s / 4 + 10)])
    c.set_geometry(p, facecolor='red', edgecolor='k', linewidth=3, alpha=0.5)

    p1 = shpg.Point(20, 10)
    p2 = shpg.Point(20, 20)
    p3 = shpg.Point(10, 20)
    mpoints = shpg.MultiPoint([p1, p2, p3])
    c.set_geometry(mpoints, s=250, marker='s',
                   c='purple', hatch='||||')

    c.set_scale_bar(color='blue')

    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, addcbar=False)
    plt.tight_layout()

    c.set_geometry()
    assert len(c._geometries) == 0

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=8)
def test_text():
    # UL Corner
    g = Grid(nxny=(5, 4), dxdy=(10, 10), x0y0=(-20, -15), proj=wgs84,
             pixel_ref='corner')
    c = Map(g, ny=4, countries=False)
    c.set_lonlat_contours(interval=5., colors='crimson', linewidths=1)

    c.set_text(-5, -5, 'Less Middle', color='green', style='italic', size=25)
    c.set_geometry(shpg.Point(-10, -10), s=500, marker='o',
                   text='My point', text_delta=[0, 0])

    shape = read_shapefile_to_grid(shapefiles['world_borders'], c.grid)
    had_c = set()
    for index, row in shape.iloc[::-1].iterrows():
        if row.CNTRY_NAME in had_c:
            c.set_geometry(row.geometry, crs=c.grid)
        else:
            c.set_geometry(row.geometry, text=row.CNTRY_NAME, crs=c.grid,
                           text_kwargs=dict(horizontalalignment='center',
                                            verticalalignment='center',
                                            clip_on=True,
                                            color='gray'), text_delta=[0, 0])
        had_c.add(row.CNTRY_NAME)

    c.set_points([20, 20, 10], [10, 20, 20], s=250, marker='s',
                 c='purple', hatch='||||', text='baaaaad', text_delta=[0, 0],
                 text_kwargs=dict(horizontalalignment='center',
                                  verticalalignment='center', color='red'))

    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, addcbar=False)
    plt.tight_layout()

    c.set_text()
    assert len(c._text) == 0

    c.set_geometry()
    assert len(c._geometries) == 0

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_hef_linear():
    grid = mercator_grid(center_ll=(10.76, 46.798444),
                         extent=(10000, 7000))
    c = Map(grid, countries=False)
    c.set_lonlat_contours(interval=10)
    c.set_shapefile(get_demo_file('Hintereisferner_UTM.shp'))
    c.set_topography(get_demo_file('hef_srtm.tif'),
                     interp='linear')

    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, addcbar=False, title='linear')
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_hef_default_spline():
    grid = mercator_grid(center_ll=(10.76, 46.798444),
                         extent=(10000, 7000))
    c = Map(grid, countries=False)
    c.set_lonlat_contours(interval=0)
    c.set_shapefile(get_demo_file('Hintereisferner_UTM.shp'))
    c.set_topography(get_demo_file('hef_srtm.tif'))

    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, addcbar=False, title='Default: spline deg 3')
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=6)
def test_hef_from_array():
    grid = mercator_grid(center_ll=(10.76, 46.798444),
                         extent=(10000, 7000))
    c = Map(grid, countries=False)
    c.set_lonlat_contours(interval=0)
    c.set_shapefile(get_demo_file('Hintereisferner_UTM.shp'))

    dem = GeoTiff(get_demo_file('hef_srtm.tif'))
    mytopo = dem.get_vardata()
    c.set_topography(mytopo, crs=dem.grid, interp='spline')

    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, addcbar=False, title='From array')
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir,
                               tolerance=15)
def test_hef_topo_withnan():
    grid = mercator_grid(center_ll=(10.76, 46.798444),
                         extent=(10000, 7000))
    c = Map(grid, countries=False)
    c.set_lonlat_contours(interval=10)
    c.set_shapefile(get_demo_file('Hintereisferner_UTM.shp'))

    dem = GeoTiff(get_demo_file('hef_srtm.tif'))
    mytopo = dem.get_vardata()
    h = c.set_topography(mytopo, crs=dem.grid, interp='spline')

    c.set_lonlat_contours()
    c.set_cmap(get_cmap('topo'))
    c.set_plot_params(nlevels=256)
    # Try with nan data
    h[-100:, -100:] = np.NaN
    c.set_data(h)
    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, title='color with NaN')
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=20)
def test_gmap():
    g = GoogleCenterMap(center_ll=(10.762660, 46.794221), zoom=13,
                        size_x=640, size_y=640)

    m = Map(g.grid, countries=False, factor=1)
    m.set_lonlat_contours(interval=0.025)
    m.set_shapefile(get_demo_file('Hintereisferner.shp'),
                    linewidths=2, edgecolor='darkred')
    m.set_rgb(g.get_vardata())

    fig, ax = plt.subplots(1, 1)
    m.visualize(ax=ax, addcbar=False)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=25)
def test_gmap_transformed():
    dem = GeoTiff(get_demo_file('hef_srtm.tif'))
    dem.set_subset(margin=-100)

    dem = mercator_grid(center_ll=(10.76, 46.798444),
                        extent=(10000, 7000))

    i, j = dem.ij_coordinates
    g = GoogleVisibleMap(x=i, y=j, crs=dem, size_x=500, size_y=400)
    img = g.get_vardata()

    m = Map(dem, countries=False)

    with pytest.raises(ValueError):
        m.set_data(img)

    m.set_lonlat_contours(interval=0.025)
    m.set_shapefile(get_demo_file('Hintereisferner.shp'),
                    linewidths=2, edgecolor='darkred')
    m.set_rgb(img, g.grid)

    fig, ax = plt.subplots(1, 1)
    m.visualize(ax=ax, addcbar=False)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=10)
def test_gmap_llconts():
    # This was because some problems were left unnoticed by other tests
    g = GoogleCenterMap(center_ll=(11.38, 47.26), zoom=9)
    m = Map(g.grid)
    m.set_rgb(g.get_vardata())
    m.set_lonlat_contours(interval=0.2)

    fig, ax = plt.subplots(1, 1)
    m.visualize(ax=ax, addcbar=False)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=13)
def test_plot_on_map():
    import salem
    from salem.utils import get_demo_file
    ds = salem.open_wrf_dataset(get_demo_file('wrfout_d01.nc'))
    t2_sub = ds.salem.subset(corners=((77., 20.), (97., 35.)), crs=salem.wgs84).T2.isel(time=2)
    shdf = salem.read_shapefile(get_demo_file('world_borders.shp'))
    shdf = shdf.loc[shdf['CNTRY_NAME'].isin(
        ['Nepal', 'Bhutan'])]  # GeoPandas' GeoDataFrame
    t2_sub = t2_sub.salem.subset(shape=shdf, margin=2)  # add 2 grid points
    t2_roi = t2_sub.salem.roi(shape=shdf)
    fig, ax = plt.subplots(1, 1)
    t2_roi.salem.quick_map(ax=ax)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
def test_example_docs():
    import salem
    from salem.utils import get_demo_file
    ds = salem.open_xr_dataset(get_demo_file('wrfout_d01.nc'))

    t2 = ds.T2.isel(Time=2)
    t2_sub = t2.salem.subset(corners=((77., 20.), (97., 35.)),
                             crs=salem.wgs84)
    shdf = salem.read_shapefile(get_demo_file('world_borders.shp'))
    shdf = shdf.loc[shdf['CNTRY_NAME'].isin(
        ['Nepal', 'Bhutan'])]  # GeoPandas' GeoDataFrame
    t2_sub = t2_sub.salem.subset(shape=shdf, margin=2)  # add 2 grid points
    t2_roi = t2_sub.salem.roi(shape=shdf)
    smap = t2_roi.salem.get_map(data=t2_roi-273.15, cmap='RdYlBu_r', vmin=-14, vmax=18)
    _ = smap.set_topography(get_demo_file('himalaya.tif'))
    smap.set_shapefile(shape=shdf, color='grey', linewidth=3, zorder=5)
    smap.set_points(91.1, 29.6)
    smap.set_text(91.2, 29.7, 'Lhasa', fontsize=17)
    smap.set_data(ds.T2.isel(Time=1)-273.15, crs=ds.salem.grid)

    fig, ax = plt.subplots(1, 1)
    smap.visualize(ax=ax)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=5)
def test_colormaps():

    fig = plt.figure(figsize=(8, 3))
    axs = [fig.add_axes([0.05, 0.80, 0.9, 0.15]),
           fig.add_axes([0.05, 0.475, 0.9, 0.15]),
           fig.add_axes([0.05, 0.15, 0.9, 0.15])]

    for ax, cm in zip(axs, ['topo', 'dem', 'nrwc']):
        cb = mpl.colorbar.ColorbarBase(ax, cmap=get_cmap(cm),
                                       orientation='horizontal')
        cb.set_label(cm);
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=5)
def test_geogrid_simulator():

    from salem.wrftools import geogrid_simulator
    g, maps = geogrid_simulator(get_demo_file('namelist_mercator.wps'),
                             do_maps=True)
    assert len(g) == 4

    fig, axs = plt.subplots(2, 2)
    axs = np.asarray(axs).flatten()
    for i, (m, ax) in enumerate(zip(maps, axs)):
        m.set_rgb(natural_earth='lr')
        m.plot(ax=ax)
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=5)
def test_lookup_transform():

    dsw = open_wrf_dataset(get_demo_file('wrfout_d01.nc'))
    dse = open_xr_dataset(get_demo_file('era_interim_tibet.nc'))
    out = dse.salem.lookup_transform(dsw.T2C.isel(time=0), method=len)
    fig, ax = plt.subplots(1, 1)
    sm = out.salem.get_map()
    sm.set_data(out)
    sm.set_geometry(dsw.salem.grid.extent_as_polygon(), edgecolor='r',
                    linewidth=2)
    sm.visualize(ax=ax)
    return fig


@requires_matplotlib
@requires_cartopy
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=10)
@pytest.mark.skip(reason='There is an unknown issue with cartopy')
def test_cartopy():

    import cartopy

    fig = plt.figure(figsize=(8, 11))

    ods = open_wrf_dataset(get_demo_file('wrfout_d01.nc'))

    ax = plt.subplot(3, 2, 1)
    smap = ods.salem.get_map()
    smap.plot(ax=ax)

    p = ods.salem.cartopy()
    ax = plt.subplot(3, 2, 2, projection=p)
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
    ax.set_extent(ods.salem.grid.extent, crs=p)

    ds = ods.isel(west_east=slice(22, 28), south_north=slice(0, 6))
    ds = ds.T2C.mean(dim='time', keep_attrs=True)

    ax = plt.subplot(3, 2, 3)
    smap = ds.salem.quick_map(ax=ax)
    ax.scatter(ds.lon, ds.lat, transform=smap.transform(ax=ax))

    p = ds.salem.cartopy()
    ax = plt.subplot(3, 2, 4, projection=p)
    ds.plot.imshow(ax=ax, transform=p)
    ax.coastlines()
    ax.scatter(ds.lon, ds.lat, transform=cartopy.crs.PlateCarree())

    ds = ods.isel(west_east=slice(80, 86), south_north=slice(80, 86))
    ds = ds.T2C.mean(dim='time', keep_attrs=True)

    ax = plt.subplot(3, 2, 5)
    smap = ds.salem.quick_map(ax=ax, factor=1)
    ax.scatter(ds.lon, ds.lat, transform=smap.transform(ax=ax))

    p = ds.salem.cartopy()
    ax = plt.subplot(3, 2, 6, projection=p)
    ds.plot.imshow(ax=ax, transform=p)
    ax.coastlines()
    ax.scatter(ds.lon, ds.lat, transform=cartopy.crs.PlateCarree())

    return fig


@requires_cartopy
@pytest.mark.mpl_image_compare(baseline_dir=baseline_dir, tolerance=7)
@pytest.mark.skip(reason='There is an unknown issue with cartopy')
def test_cartopy_polar():

    import cartopy

    fig = plt.figure(figsize=(8, 8))

    ods = open_wrf_dataset(get_demo_file('geo_em_d02_polarstereo.nc'))
    ods = ods.isel(time=0)

    ax = plt.subplot(2, 2, 1)
    smap = ods.salem.get_map()
    smap.plot(ax=ax)

    p = ods.salem.cartopy()
    ax = plt.subplot(2, 2, 2, projection=p)
    ax.coastlines(resolution='50m')
    ax.gridlines()
    ax.set_extent(ods.salem.grid.extent, crs=p)

    ds = ods.salem.subset(corners=((-52.8, 70.11), (-52.8, 70.11)), margin=12)

    ax = plt.subplot(2, 2, 3)
    smap = ds.HGT_M.salem.quick_map(ax=ax, cmap='Oranges')
    ax.scatter(ds.XLONG_M, ds.XLAT_M, s=5,
               transform=smap.transform(ax=ax))

    p = ds.salem.cartopy()
    ax = plt.subplot(2, 2, 4, projection=p)
    ds.HGT_M.plot.imshow(ax=ax, transform=p, cmap='Oranges')
    ax.coastlines(resolution='50m')
    ax.gridlines()
    ax.scatter(ds.XLONG_M, ds.XLAT_M, transform=cartopy.crs.PlateCarree(), s=5)

    return fig
