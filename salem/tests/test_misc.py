from __future__ import division

import unittest
import shutil
import os
import time
import warnings
import copy

import pytest
import netCDF4
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from salem.tests import requires_travis, requires_geopandas, \
    requires_matplotlib, requires_xarray
from salem import utils, transform_geopandas, GeoTiff, read_shapefile, sio
from salem import read_shapefile_to_grid, graphics, Grid, mercator_grid, wgs84
from salem.utils import get_demo_file

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass

current_dir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(current_dir, 'tmp')
if not os.path.exists(testdir):
    os.makedirs(testdir)


@requires_geopandas
def create_dummy_shp(fname):

    import shapely.geometry as shpg
    import geopandas as gpd

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
    df.to_file(of)
    return of


def delete_test_dir():
    if os.path.exists(testdir):
        shutil.rmtree(testdir)


class TestUtils(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(testdir):
            os.makedirs(testdir)

    def tearDown(self):
        delete_test_dir()

    @requires_travis
    def test_empty_cache(self):

        utils.empty_cache()

    def test_demofiles(self):

        self.assertTrue(os.path.exists(utils.get_demo_file('dem_wgs84.nc')))
        self.assertTrue(utils.get_demo_file('dummy') is None)

    def test_read_colormap(self):

        cl = utils.read_colormap('topo') * 256
        assert_allclose(cl[4, :], (177, 242, 196))
        assert_allclose(cl[-1, :], (235, 233, 235))

        cl = utils.read_colormap('dem') * 256
        assert_allclose(cl[4, :], (153,100, 43))
        assert_allclose(cl[-1, :], (255,255,255))


class TestIO(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(testdir):
            os.makedirs(testdir)

    def tearDown(self):
        delete_test_dir()

    @requires_geopandas
    def test_cache_working(self):

        f1 = 'f1.shp'
        f1 = create_dummy_shp(f1)
        cf1 = utils.cached_shapefile_path(f1)
        self.assertFalse(os.path.exists(cf1))
        _ = read_shapefile(f1)
        self.assertFalse(os.path.exists(cf1))
        _ = read_shapefile(f1, cached=True)
        self.assertTrue(os.path.exists(cf1))
        # nested calls
        self.assertTrue(cf1 == utils.cached_shapefile_path(cf1))

        # wait a bit
        time.sleep(0.1)
        f1 = create_dummy_shp(f1)
        cf2 = utils.cached_shapefile_path(f1)
        self.assertFalse(os.path.exists(cf1))
        _ = read_shapefile(f1, cached=True)
        self.assertFalse(os.path.exists(cf1))
        self.assertTrue(os.path.exists(cf2))
        df = read_shapefile(f1, cached=True)
        np.testing.assert_allclose(df.min_x, [1., 2.])
        np.testing.assert_allclose(df.max_x, [2., 3.])
        np.testing.assert_allclose(df.min_y, [1., 1.3])
        np.testing.assert_allclose(df.max_y, [2., 2.3])

        self.assertRaises(ValueError, read_shapefile, 'f1.sph')
        self.assertRaises(ValueError, utils.cached_shapefile_path, 'f1.splash')


    @requires_geopandas
    def test_read_to_grid(self):

        g = GeoTiff(utils.get_demo_file('hef_srtm.tif'))
        sf = utils.get_demo_file('Hintereisferner_UTM.shp')

        df1 = read_shapefile_to_grid(sf, g.grid)

        df2 = transform_geopandas(read_shapefile(sf), to_crs=g.grid)
        assert_allclose(df1.geometry[0].exterior.coords,
                        df2.geometry[0].exterior.coords)


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
        g = Grid(nxny=(5, 4), dxdy=(1, 1), ll_corner=(0, 0), proj=wgs84,
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
        g = Grid(nxny=(5, 4), dxdy=(1, 1), ll_corner=(0.5, 0.5), proj=wgs84,
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
        rgb1 = rgb1[np.min(pok[0]):np.max(pok[0]),
                    np.min(pok[1]):np.max(pok[1]),...]
        rgb2 = rgb2[np.min(pok[0]):np.max(pok[0]),
                    np.min(pok[1]):np.max(pok[1]),...]
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
        grid = mercator_grid(center_ll=(-20, 40),
                                        extent=(2000, 2000), nx=10)
        c = graphics.Map(grid)

        # Assigning wrongly shaped data should, however
        self.assertRaises(ValueError, c.set_data, np.zeros((3, 8)))


class TestSkyIsFalling(unittest.TestCase):

    @requires_matplotlib
    def test_projplot(self):

        # this caused many problems on fabien's laptop.
        # this is just to be sure that on your system, everything is fine

        import pyproj
        import matplotlib.pyplot as plt

        wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
        fig = plt.figure()
        plt.close()

        srs = '+units=m +proj=lcc +lat_1=29.0 +lat_2=29.0 +lat_0=29.0 +lon_0=89.8'

        proj_out = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
        proj_in = pyproj.Proj(srs, preserve_units=True)

        lon, lat = pyproj.transform(proj_in, proj_out, -2235000, -2235000)
        np.testing.assert_allclose(lon, 70.75731, atol=1e-5)


class TestXarray(unittest.TestCase):

    @requires_xarray
    def test_era(self):

        ds = sio.open_xr_dataset(get_demo_file('era_interim_tibet.nc'))
        self.assertEqual(ds.salem.x_dim, 'longitude')
        self.assertEqual(ds.salem.y_dim, 'latitude')

        lon = 91.1
        lat = 31.1
        dss = ds.salem.subset(corners=((lon, lat), (lon, lat)), margin=1)

        self.assertEqual(len(dss.latitude), 3)
        self.assertEqual(len(dss.longitude), 3)

        np.testing.assert_almost_equal(dss.longitude, [90.0, 90.75, 91.5])

    @requires_xarray
    @requires_geopandas  # because of the grid tests, more robust with GDAL
    def test_basic_wrf(self):
        import xarray as xr

        ds = sio.open_xr_dataset(get_demo_file('wrf_tip_d1.nc'))

        # this is because read_dataset changes some stuff, let's see if
        # georef still ok
        dsxr = xr.open_dataset(get_demo_file('wrf_tip_d1.nc'))
        assert ds.salem.grid == dsxr.salem.grid

        lon, lat = ds.salem.grid.ll_coordinates
        assert_allclose(lon, ds['XLONG'], atol=1e-4)
        assert_allclose(lat, ds['XLAT'], atol=1e-4)

        # then something strange happened
        assert ds.isel(Time=0).salem.grid == ds.salem.grid
        assert ds.isel(Time=0).T2.salem.grid == ds.salem.grid

        nlon, nlat = ds.isel(Time=0).T2.salem.grid.ll_coordinates
        assert_allclose(nlon, ds['XLONG'], atol=1e-4)
        assert_allclose(nlat, ds['XLAT'], atol=1e-4)

        # the grid should not be missunderstood as lonlat
        t2 = ds.T2.isel(Time=0) - 273.15
        with pytest.raises(RuntimeError):
            g = t2.salem.grid

    @requires_xarray
    @requires_geopandas  # because of the grid tests, more robust with GDAL
    def test_wrf(self):
        import xarray as xr

        ds = sio.open_wrf_dataset(get_demo_file('wrf_tip_d1.nc'))

        # this is because read_dataset changes some stuff, let's see if
        # georef still ok
        dsxr = xr.open_dataset(get_demo_file('wrf_tip_d1.nc'))
        assert ds.salem.grid == dsxr.salem.grid

        lon, lat = ds.salem.grid.ll_coordinates
        assert_allclose(lon, ds['lon'], atol=1e-4)
        assert_allclose(lat, ds['lat'], atol=1e-4)

        # then something strange happened
        assert ds.isel(time=0).salem.grid == ds.salem.grid
        assert ds.isel(time=0).T2.salem.grid == ds.salem.grid

        nlon, nlat = ds.isel(time=0).T2.salem.grid.ll_coordinates
        assert_allclose(nlon, ds['lon'], atol=1e-4)
        assert_allclose(nlat, ds['lat'], atol=1e-4)

        # the grid should not be missunderstood as lonlat
        t2 = ds.T2.isel(time=0) - 273.15
        with pytest.raises(RuntimeError):
            g = t2.salem.grid

    @requires_xarray
    def test_diagvars(self):

        wf = get_demo_file('wrf_cropped.nc')
        ncl_out = get_demo_file('wrf_cropped_ncl.nc')

        w = sio.open_wrf_dataset(wf)
        nc = sio.open_xr_dataset(ncl_out)

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=1e-6)

        ref = nc['SLP']
        tot = w['SLP']
        tot = tot.values
        assert_allclose(ref, tot, rtol=1e-6)

        w = w.isel(time=1, south_north=slice(12, 16), west_east=slice(9, 16))
        nc = nc.isel(Time=1, south_north=slice(12, 16), west_east=slice(9, 16))

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=1e-6)

        ref = nc['SLP']
        tot = w['SLP']
        tot = tot.values
        assert_allclose(ref, tot, rtol=1e-6)

        w = w.isel(bottom_top=slice(3, 5))
        nc = nc.isel(bottom_top=slice(3, 5))

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=1e-6)

        ref = nc['SLP']
        tot = w['SLP']
        tot = tot.values
        assert_allclose(ref, tot, rtol=1e-6)

    @requires_xarray
    def test_unstagger(self):

        wf = get_demo_file('wrf_cropped.nc')

        w = sio.open_wrf_dataset(wf)
        nc = sio.open_xr_dataset(wf)

        nc['PH_UNSTAGG'] = nc['P']*0.
        uns = nc['PH'].isel(bottom_top_stag=slice(0, -1)).values + \
              nc['PH'].isel(bottom_top_stag=slice(1, len(nc.bottom_top_stag))).values
        nc['PH_UNSTAGG'].values = uns * 0.5

        assert_allclose(w['PH'], nc['PH_UNSTAGG'])

        wn = w.isel(west_east=slice(4, 8))
        ncn = nc.isel(west_east=slice(4, 8))
        assert_allclose(wn['PH'], ncn['PH_UNSTAGG'])

        wn = w.isel(south_north=slice(4, 8), time=1)
        ncn = nc.isel(south_north=slice(4, 8), Time=1)
        assert_allclose(wn['PH'], ncn['PH_UNSTAGG'])

        wn = w.isel(west_east=4)
        ncn = nc.isel(west_east=4)
        assert_allclose(wn['PH'], ncn['PH_UNSTAGG'])

        wn = w.isel(bottom_top=4)
        ncn = nc.isel(bottom_top=4)
        assert_allclose(wn['PH'], ncn['PH_UNSTAGG'])

        wn = w.isel(bottom_top=0)
        ncn = nc.isel(bottom_top=0)
        assert_allclose(wn['PH'], ncn['PH_UNSTAGG'])

        wn = w.isel(bottom_top=-1)
        ncn = nc.isel(bottom_top=-1)
        assert_allclose(wn['PH'], ncn['PH_UNSTAGG'])

    @requires_xarray
    def test_prcp(self):

        wf = get_demo_file('wrfout_d01.nc')

        w = sio.open_wrf_dataset(wf)
        nc = sio.open_xr_dataset(wf)

        nc['REF_PRCP_NC'] = nc['RAINNC']*0.
        uns = nc['RAINNC'].isel(Time=slice(1, len(nc.bottom_top_stag))).values - \
              nc['RAINNC'].isel(Time=slice(0, -1)).values
        nc['REF_PRCP_NC'].values[1:, ...] = uns * 60 / 180.  # for three hours
        nc['REF_PRCP_NC'].values[0, ...] = np.NaN

        nc['REF_PRCP_C'] = nc['RAINC']*0.
        uns = nc['RAINC'].isel(Time=slice(1, len(nc.bottom_top_stag))).values - \
              nc['RAINC'].isel(Time=slice(0, -1)).values
        nc['REF_PRCP_C'].values[1:, ...] = uns * 60 / 180.  # for three hours
        nc['REF_PRCP_C'].values[0, ...] = np.NaN

        nc['REF_PRCP'] = nc['REF_PRCP_C'] + nc['REF_PRCP_NC']

        for suf in ['_NC', '_C', '']:

            assert_allclose(w['PRCP' + suf], nc['REF_PRCP' + suf], rtol=1e-5)

            wn = w.isel(time=slice(1, 3))
            ncn = nc.isel(Time=slice(1, 3))
            assert_allclose(wn['PRCP' + suf], ncn['REF_PRCP' + suf], rtol=1e-5)

            wn = w.isel(time=2)
            ncn = nc.isel(Time=2)
            assert_allclose(wn['PRCP' + suf], ncn['REF_PRCP' + suf], rtol=1e-5)

            wn = w.isel(time=1)
            ncn = nc.isel(Time=1)
            assert_allclose(wn['PRCP' + suf], ncn['REF_PRCP' + suf], rtol=1e-5)

            wn = w.isel(time=0)
            self.assertTrue(~np.any(np.isfinite(wn['PRCP' + suf].values)))

            wn = w.isel(time=slice(1, 3), south_north=slice(50, -1))
            ncn = nc.isel(Time=slice(1, 3), south_north=slice(50, -1))
            assert_allclose(wn['PRCP' + suf], ncn['REF_PRCP' + suf], rtol=1e-5)

            wn = w.isel(time=2, south_north=slice(50, -1))
            ncn = nc.isel(Time=2, south_north=slice(50, -1))
            assert_allclose(wn['PRCP' + suf], ncn['REF_PRCP' + suf], rtol=1e-5)

            wn = w.isel(time=1, south_north=slice(50, -1))
            ncn = nc.isel(Time=1, south_north=slice(50, -1))
            assert_allclose(wn['PRCP' + suf], ncn['REF_PRCP' + suf], rtol=1e-5)

            wn = w.isel(time=0, south_north=slice(50, -1))
            self.assertTrue(~np.any(np.isfinite(wn['PRCP' + suf].values)))

    @requires_xarray
    @requires_geopandas  # because of the grid tests, more robust with GDAL
    def test_transform_logic(self):

        # This is just for the naming and dim logic, the rest is tested elsewh
        ds1 = sio.open_wrf_dataset(get_demo_file('wrfout_d01.nc'))
        ds2 = sio.open_wrf_dataset(get_demo_file('wrfout_d01.nc'))

        # 2darray case
        t2 = ds2.T2.isel(time=1)
        with pytest.raises(ValueError):
            ds1.salem.transform_and_add(t2.values, grid=t2.salem.grid)

        ds1.salem.transform_and_add(t2.values, grid=t2.salem.grid, name='t2_2darr')
        assert 't2_2darr' in ds1
        assert_allclose(ds1.t2_2darr.coords['south_north'],
                        t2.coords['south_north'])
        assert_allclose(ds1.t2_2darr.coords['west_east'],
                        t2.coords['west_east'])
        assert ds1.salem.grid == ds1.t2_2darr.salem.grid

        # 3darray case
        t2 = ds2.T2
        ds1.salem.transform_and_add(t2.values, grid=t2.salem.grid, name='t2_3darr')
        assert 't2_3darr' in ds1
        assert_allclose(ds1.t2_3darr.coords['south_north'],
                        t2.coords['south_north'])
        assert_allclose(ds1.t2_3darr.coords['west_east'],
                        t2.coords['west_east'])
        assert 'time' in ds1.t2_3darr.coords

        # dataarray case
        ds1.salem.transform_and_add(t2, name='NEWT2')
        assert 'NEWT2' in ds1
        assert_allclose(ds1.NEWT2, ds1.T2)
        assert_allclose(ds1.t2_3darr.coords['south_north'],
                        t2.coords['south_north'])
        assert_allclose(ds1.t2_3darr.coords['west_east'],
                        t2.coords['west_east'])
        assert 'time' in ds1.t2_3darr.coords

        # dataset case
        ds1.salem.transform_and_add(ds2[['RAINC', 'RAINNC']],
                                    name={'RAINC':'PRCPC',
                                          'RAINNC': 'PRCPNC'})
        assert 'PRCPC' in ds1
        assert_allclose(ds1.PRCPC, ds1.RAINC)
        assert 'time' in ds1.PRCPNC.coords

        # what happens with external data?
        dse = sio.open_xr_dataset(get_demo_file('era_interim_tibet.nc'))
        out = ds1.salem.transform(dse.t2m, interp='linear')
        assert_allclose(out.coords['south_north'],
                        t2.coords['south_north'])
        assert_allclose(out.coords['west_east'],
                        t2.coords['west_east'])

    @requires_xarray
    def test_full_wrf_wfile(self):

        from salem.wrftools import var_classes

        # TODO: these tests are qualitative and should be compared against ncl
        f = get_demo_file('wrf_d01_allvars_cropped.nc')
        ds = sio.open_wrf_dataset(f)

        # making a repr was causing trouble because of the small chunks
        _ = ds.__repr__()

        # just check that the data is here
        var_classes = copy.deepcopy(var_classes)
        for vn in var_classes:
            _ = ds[vn].values
            dss = ds.isel(west_east=slice(2, 6), south_north=slice(2, 5),
                          bottom_top=slice(0, 15))
            _ = dss[vn].values
            dss = ds.isel(west_east=1, south_north=2,
                          bottom_top=3,  time=2)
            _ = dss[vn].values


class TestGeogridSim(unittest.TestCase):

    @requires_geopandas
    def test_lambert(self):

        from salem.wrftools import geogrid_simulator

        g, m = geogrid_simulator(get_demo_file('namelist_lambert.wps'))

        assert len(g) == 3

        for i in [1, 2, 3]:
            fg = get_demo_file('geo_em_d0{}_lambert.nc'.format(i))
            ds = netCDF4.Dataset(fg)
            lon, lat = g[i-1].ll_coordinates
            assert_allclose(lon, ds['XLONG_M'][0, ...], atol=1e-4)
            assert_allclose(lat, ds['XLAT_M'][0, ...], atol=1e-4)

    @requires_geopandas
    def test_mercator(self):

        from salem.wrftools import geogrid_simulator

        g, m = geogrid_simulator(get_demo_file('namelist_mercator.wps'))

        assert len(g) == 4

        for i in [1, 2, 3, 4]:
            fg = get_demo_file('geo_em_d0{}_mercator.nc'.format(i))
            ds = netCDF4.Dataset(fg)
            lon, lat = g[i-1].ll_coordinates
            assert_allclose(lon, ds['XLONG_M'][0, ...], atol=1e-4)
            assert_allclose(lat, ds['XLAT_M'][0, ...], atol=1e-4)