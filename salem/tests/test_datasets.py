from __future__ import division

import unittest
import warnings
from datetime import datetime


import numpy as np
import netCDF4


import pandas as pd
import xarray as xr

from numpy.testing import assert_array_equal, assert_allclose
from salem import Grid
from salem.utils import get_demo_file
from salem import wgs84
from salem import wrftools, mercator_grid
from salem.datasets import (GeoDataset, GeoNetcdf, GeoTiff, WRF,
                            GoogleCenterMap, GoogleVisibleMap, EsriITMIX)
from salem.tests import (requires_rasterio, requires_motionless,
                         requires_geopandas, requires_internet,
                         requires_matplotlib, requires_shapely)


class TestDataset(unittest.TestCase):

    def test_period(self):
        """See if simple operations work well"""

        g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        d = GeoDataset(g)
        self.assertTrue(d.time is None)
        self.assertTrue(d.sub_t is None)
        self.assertTrue(d.t0 is None)
        self.assertTrue(d.t1 is None)

        t = pd.date_range('1/1/2011', periods=72, freq='D')
        d = GeoDataset(g, time=t)
        assert_array_equal(d.time, t)
        assert_array_equal(d.sub_t, [0, 71])
        assert_array_equal(d.t0, t[0])
        assert_array_equal(d.t1, t[-1])
        d.set_period(t0='2011-01-03')
        assert_array_equal(d.sub_t, [2, 71])
        assert_array_equal(d.t0, t[2])
        assert_array_equal(d.t1, t[-1])
        d.set_period(t0='2011-01-03', t1=datetime(2011, 1, 5))
        assert_array_equal(d.sub_t, [2, 4])
        assert_array_equal(d.t0, t[2])
        assert_array_equal(d.t1, t[4])
        d.set_period(t1=datetime(2011, 1, 5))
        assert_array_equal(d.sub_t, [0, 4])
        assert_array_equal(d.t0, t[0])
        assert_array_equal(d.t1, t[4])
        d = GeoDataset(g, time=pd.Series(t, index=t))
        assert_array_equal(d.time, t)
        d.set_period(t0='2011-01-03', t1=datetime(2011, 1, 5))
        assert_array_equal(d.sub_t, [2, 4])
        assert_array_equal(d.t0, t[2])
        assert_array_equal(d.t1, t[4])
        d.set_period()
        assert_array_equal(d.time, t)
        assert_array_equal(d.sub_t, [0, 71])
        assert_array_equal(d.t0, t[0])
        assert_array_equal(d.t1, t[-1])

        self.assertRaises(NotImplementedError, d.get_vardata)

    @requires_rasterio
    @requires_shapely
    def test_subset(self):
        """See if simple operations work well"""

        import shapely.geometry as shpg

        g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        d = GeoDataset(g)
        self.assertTrue(isinstance(d, GeoDataset))
        self.assertEqual(g, d.grid)

        d.set_subset(corners=([0, 0], [2, 2]), crs=wgs84)
        self.assertEqual(g, d.grid)

        d.set_subset()
        self.assertEqual(g, d.grid)

        d.set_subset(margin=-1)
        lon, lat = d.grid.ll_coordinates
        self.assertEqual(lon, 1)
        self.assertEqual(lat, 1)

        d.set_subset(corners=([0.1, 0.1], [1.9, 1.9]), crs=wgs84)
        self.assertEqual(g, d.grid)

        d.set_subset(corners=([0.51, 0.51], [1.9, 1.9]), crs=wgs84)
        self.assertNotEqual(g, d.grid)

        gm = Grid(nxny=(1, 1), dxdy=(1, 1), x0y0=(1, 1), proj=wgs84)
        d.set_subset(corners=([1, 1], [1, 1]), crs=wgs84)
        self.assertEqual(gm, d.grid)

        d.set_subset()
        d.set_roi()
        d.set_roi(corners=([1, 1], [1, 1]), crs=wgs84)
        d.set_subset(toroi=True)
        self.assertEqual(gm, d.grid)

        gm = Grid(nxny=(1, 1), dxdy=(1, 1), x0y0=(2, 2), proj=wgs84)
        d.set_subset(corners=([2, 2], [2, 2]), crs=wgs84)
        self.assertEqual(gm, d.grid)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            d.set_subset(corners=([-4, -4], [5, 5]), crs=wgs84)
            self.assertEqual(g, d.grid)
            # Verify some things
            assert len(w) >= 2

        self.assertRaises(RuntimeError, d.set_subset, corners=([-1, -1],
                                                               [-1, -1]))
        self.assertRaises(RuntimeError, d.set_subset, corners=([5, 5],
                                                               [5, 5]))

        shpf = get_demo_file('Hintereisferner.shp')
        reff = get_demo_file('hef_roi.tif')
        d = GeoTiff(reff)
        d.set_roi(shape=shpf)
        ref = d.get_vardata()
        # same errors as IDL: ENVI is just wrong
        self.assertTrue(np.sum(ref != d.roi) < 9)

        g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84,
                 pixel_ref='corner')
        p = shpg.Polygon([(1.5, 1.), (2., 1.5), (1.5, 2.), (1., 1.5)])
        roi = g.region_of_interest(geometry=p)
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,0]], roi)

        d = GeoDataset(g)
        d.set_roi(corners=([1.1,1.1], [1.9,1.9]))
        d.set_subset(toroi=True)
        np.testing.assert_array_equal([[1]], d.roi)
        d.set_subset()
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,0]], d.roi)
        d.set_roi()
        np.testing.assert_array_equal([[0,0,0],[0,0,0],[0,0,0]], d.roi)

        # Raises
        self.assertRaises(RuntimeError, d.set_subset, toroi=True)


class TestGeotiff(unittest.TestCase):

    @requires_rasterio
    def test_subset(self):
        """Open geotiff, do subsets and stuff"""
        go = get_demo_file('hef_srtm.tif')
        gs = get_demo_file('hef_srtm_subset.tif')

        go = GeoTiff(go)
        gs = GeoTiff(gs)

        go.set_roi(grid=gs.grid)
        go.set_subset(toroi=True)
        ref = gs.get_vardata()
        totest = go.get_vardata()
        np.testing.assert_array_equal(ref.shape, (go.grid.ny, go.grid.nx))
        np.testing.assert_array_equal(ref.shape, totest.shape)
        np.testing.assert_array_equal(ref, totest)
        go.set_roi()
        go.set_subset()

        eps = 1e-5
        ex = gs.grid.extent_in_crs(crs=wgs84) # [left, right, bot, top
        go.set_subset(corners=((ex[0], ex[2]+eps), (ex[1], ex[3]-eps)),
                      crs=wgs84,
                      margin=-2)
        ref = gs.get_vardata()[2:-2, 2:-2]
        totest = go.get_vardata()
        np.testing.assert_array_equal(ref.shape, totest.shape)
        np.testing.assert_array_equal(ref, totest)
        go.set_roi()
        go.set_subset()

    @requires_rasterio
    def test_itmix(self):

        gf = get_demo_file('02_surface_Academy_1997_UTM47.asc')
        ds = EsriITMIX(gf)
        topo = ds.get_vardata()

    @requires_rasterio
    def test_xarray(self):

        from salem import open_xr_dataset

        go = get_demo_file('hef_srtm.tif')
        gs = get_demo_file('hef_srtm_subset.tif')

        geo = GeoTiff(go)
        go = open_xr_dataset(go)
        gs = open_xr_dataset(gs)

        gos = go.salem.subset(grid=gs.salem.grid)

        ref = gs['data']
        totest = gos['data']
        np.testing.assert_array_equal(ref.shape, (gos.salem.grid.ny, gos.salem.grid.nx))
        np.testing.assert_array_equal(ref.shape, totest.shape)
        np.testing.assert_array_equal(ref, totest)
        rlon, rlat = geo.grid.ll_coordinates
        tlon, tlat = go.salem.grid.ll_coordinates
        assert_allclose(rlon, tlon)
        assert_allclose(rlat, tlat)


class TestGeoNetcdf(unittest.TestCase):

    def test_eraint(self):

        f = get_demo_file('era_interim_tibet.nc')
        d = GeoNetcdf(f)
        assert d.grid.origin == 'upper-left'

        stat_lon = 91.1
        stat_lat = 31.1
        with netCDF4.Dataset(f) as nc:
            nc.set_auto_mask(False)
            flon = nc.variables['longitude'][:]
            flat = nc.variables['latitude'][:]
            alon = np.argmin(np.abs(flon - stat_lon))
            alat = np.argmin(np.abs(flat - stat_lat))

            d.set_subset(corners=((stat_lon, stat_lat), (stat_lon, stat_lat)))
            slon, slat = d.grid.ll_coordinates
            assert_array_equal(flon[alon], slon)
            assert_allclose(flat[alat], slat)
            # Exotic subset
            assert_array_equal(flon[alon], d.get_vardata('longitude'))
            assert_allclose(flat[alat], d.get_vardata('latitude'))

            assert_allclose(nc.variables['t2m'][:, alat, alon],
                            np.squeeze(d.get_vardata('t2m')))

            d.set_period(t0='2012-06-01 06:00:00', t1='2012-06-01 12:00:00')
            assert_allclose(nc.variables['t2m'][1:3, alat, alon],
                            np.squeeze(d.get_vardata('t2m')))

    def test_as_xarray(self):

        f = get_demo_file('era_interim_tibet.nc')
        d = GeoNetcdf(f)
        t2 = d.get_vardata('t2m', as_xarray=True)

        stat_lon = 91.1
        stat_lat = 31.1
        d.set_subset(corners=((stat_lon, stat_lat), (stat_lon, stat_lat)))
        t2_sub = d.get_vardata('t2m', as_xarray=True)
        np.testing.assert_allclose(t2_sub - t2, np.zeros(4).reshape((4,1,1)))

        d.set_period(t0='2012-06-01 06:00:00', t1='2012-06-01 12:00:00')
        t2_sub = d.get_vardata('t2m', as_xarray=True)
        np.testing.assert_allclose(t2_sub - t2, np.zeros(2).reshape((2,1,1)))

        wf = get_demo_file('wrf_cropped.nc')
        d = WRF(wf)
        tk = d.get_vardata('TK', as_xarray=True)
        # TODO: the z dim is not ok

    @requires_geopandas
    def test_wrf(self):
        """Open WRF, do subsets and stuff"""

        fs = get_demo_file('chinabang.shp')

        for d in ['1', '2']:
            fw = get_demo_file('wrf_tip_d{}.nc'.format(d))
            d = GeoNetcdf(fw)
            self.assertTrue(isinstance(d, GeoDataset))
            mylon, mylat = d.grid.ll_coordinates
            reflon = d.get_vardata('XLONG')
            reflat = d.get_vardata('XLAT')
            np.testing.assert_allclose(reflon, mylon, rtol=0.000001)
            np.testing.assert_allclose(reflat, mylat, rtol=0.00001)

            d.set_roi(shape=fs)
            np.testing.assert_array_equal(d.get_vardata('roi'), d.roi)

        d1 = GeoNetcdf(get_demo_file('wrf_tip_d1.nc'))
        d2 = GeoNetcdf(get_demo_file('wrf_tip_d2.nc'))

        # Auto dimensions
        self.assertTrue(d1.t_dim == 'Time')
        self.assertTrue(d1.x_dim == 'west_east')
        self.assertTrue(d1.y_dim == 'south_north')
        self.assertTrue(d1.z_dim is None)

        #Time
        assert_array_equal(d1.time, pd.to_datetime([datetime(2005, 9, 21),
                                                    datetime(2005, 9, 21, 3)]))

        assert_array_equal(d2.time, pd.to_datetime([datetime(2005, 9, 21),
                                                    datetime(2005, 9, 21, 1)]))
        bef = d2.get_vardata('T2')
        d2.set_period(t0=datetime(2005, 9, 21, 1))
        assert_array_equal(bef[[1], ...], d2.get_vardata('T2'))
        d2.set_period()
        assert_array_equal(bef, d2.get_vardata('T2'))
        d2.set_period(t1=datetime(2005, 9, 21, 0))
        assert_array_equal(bef[[0], ...], d2.get_vardata('T2'))

        # ROIS
        d1.set_roi(grid=d2.grid)
        d1.set_subset(toroi=True)
        self.assertEqual(d1.grid.nx * 3, d2.grid.nx)
        self.assertEqual(d1.grid.ny * 3, d2.grid.ny)
        self.assertTrue(np.min(d1.roi) == 1)

        mylon, mylat = d1.grid.ll_coordinates
        reflon = d1.get_vardata('XLONG')
        reflat = d1.get_vardata('XLAT')
        np.testing.assert_allclose(reflon, mylon, atol=1e-4)
        np.testing.assert_allclose(reflat, mylat, atol=1e-4)

        reflon = d2.get_vardata('XLONG')[1::3, 1::3]
        reflat = d2.get_vardata('XLAT')[1::3, 1::3]
        np.testing.assert_allclose(reflon, mylon, atol=1e-4)
        np.testing.assert_allclose(reflat, mylat, atol=1e-4)

        # Mercator
        d = GeoNetcdf(get_demo_file('wrf_mercator.nc'))
        mylon, mylat = d.grid.ll_coordinates
        reflon = np.squeeze(d.get_vardata('XLONG'))
        reflat = np.squeeze(d.get_vardata('XLAT'))
        np.testing.assert_allclose(reflon, mylon, atol=1e-5)
        np.testing.assert_allclose(reflat, mylat, atol=1e-5)

        # Test xarray
        ds = xr.open_dataset(get_demo_file('wrf_mercator.nc'))
        mylon, mylat = ds.salem.grid.ll_coordinates
        np.testing.assert_allclose(reflon, mylon, atol=1e-5)
        np.testing.assert_allclose(reflat, mylat, atol=1e-5)
        d = GeoNetcdf(get_demo_file('wrf_tip_d1.nc'))
        reflon = np.squeeze(d.get_vardata('XLONG'))
        reflat = np.squeeze(d.get_vardata('XLAT'))
        ds = xr.open_dataset(get_demo_file('wrf_tip_d1.nc'))
        mylon, mylat = ds.salem.grid.ll_coordinates
        np.testing.assert_allclose(reflon, mylon, atol=1e-4)
        np.testing.assert_allclose(reflat, mylat, atol=1e-4)

    @requires_geopandas
    def test_wrf_polar(self):

        d = GeoNetcdf(get_demo_file('geo_em_d01_polarstereo.nc'))
        mylon, mylat = d.grid.ll_coordinates
        reflon = np.squeeze(d.get_vardata('XLONG_M'))
        reflat = np.squeeze(d.get_vardata('XLAT_M'))

        np.testing.assert_allclose(reflon, mylon, atol=5e-3)
        np.testing.assert_allclose(reflat, mylat, atol=5e-3)

        d = GeoNetcdf(get_demo_file('geo_em_d02_polarstereo.nc'))
        mylon, mylat = d.grid.ll_coordinates
        reflon = np.squeeze(d.get_vardata('XLONG_M'))
        reflat = np.squeeze(d.get_vardata('XLAT_M'))

        np.testing.assert_allclose(reflon, mylon, atol=1e-4)
        np.testing.assert_allclose(reflat, mylat, atol=1e-4)

    @requires_geopandas
    def test_wrf_latlon(self):

        d = GeoNetcdf(get_demo_file('geo_em.d01_lon-lat.nc'))
        mylon, mylat = d.grid.ll_coordinates
        reflon = np.squeeze(d.get_vardata('XLONG_M'))
        reflat = np.squeeze(d.get_vardata('XLAT_M'))

        np.testing.assert_allclose(reflon, mylon, atol=1e-4)
        np.testing.assert_allclose(reflat, mylat, atol=1e-4)

        d = GeoNetcdf(get_demo_file('geo_em.d04_lon-lat.nc'))
        mylon, mylat = d.grid.ll_coordinates
        reflon = np.squeeze(d.get_vardata('XLONG_M'))
        reflat = np.squeeze(d.get_vardata('XLAT_M'))

        np.testing.assert_allclose(reflon, mylon, atol=1e-4)
        np.testing.assert_allclose(reflat, mylat, atol=1e-4)

    def test_longtime(self):
        """There was a bug with time"""

        fs = get_demo_file('test_longtime.nc')
        c = GeoNetcdf(fs)
        self.assertEqual(len(c.time), 2424)
        assert_array_equal(c.time[0:2], pd.to_datetime([datetime(1801, 10, 1),
                                                        datetime(1801, 11,
                                                                 1)]))

    def test_diagnostic_vars(self):

        d = WRF(get_demo_file('wrf_tip_d1.nc'))
        d2 = GeoNetcdf(get_demo_file('wrf_tip_d2.nc'))
        self.assertTrue('T2C' in d.variables)

        ref = d.get_vardata('T2')
        tot = d.get_vardata('T2C') + 273.15
        np.testing.assert_allclose(ref, tot)

        d.set_roi(grid=d2.grid)
        d.set_subset(toroi=True)
        ref = d.get_vardata('T2')
        tot = d.get_vardata('T2C') + 273.15

        self.assertEqual(tot.shape[-1] * 3, d2.grid.nx)
        self.assertEqual(tot.shape[-2] * 3, d2.grid.ny)
        np.testing.assert_allclose(ref, tot)

        d = WRF(get_demo_file('wrf_tip_d1.nc'))
        ref = d.variables['T2'][:]
        d.set_subset(margin=-5)
        tot = d.get_vardata('T2')
        assert_array_equal(ref.shape[1]-10, tot.shape[1])
        assert_array_equal(ref.shape[2]-10, tot.shape[2])
        assert_array_equal(ref[:, 5:-5, 5:-5], tot)


class TestGoogleStaticMap(unittest.TestCase):

    @requires_internet
    @requires_motionless
    @requires_matplotlib
    def test_center(self):
        import matplotlib as mpl
        gm = GoogleCenterMap(center_ll=(10.762660, 46.794221), zoom=13,
                             size_x=500, size_y=500, use_cache=False)
        gm.set_roi(shape=get_demo_file('Hintereisferner.shp'))
        gm.set_subset(toroi=True, margin=10)
        img = gm.get_vardata()[..., :3]
        img[np.nonzero(gm.roi == 0)] /= 2.

        # from PIL import Image
        # Image.fromarray((img * 255).astype(np.uint8)).save(
        #     get_demo_file('hef_google_roi.png'))
        ref = mpl.image.imread(get_demo_file('hef_google_roi.png'))
        rmsd = np.sqrt(np.mean((ref - img)**2))
        self.assertTrue(rmsd < 0.2)
        # assert_allclose(ref, img, atol=2e-2)

        gm = GoogleCenterMap(center_ll=(10.762660, 46.794221), zoom=13,
                             size_x=500, size_y=500)
        gm.set_roi(shape=get_demo_file('Hintereisferner.shp'))
        gm.set_subset(toroi=True, margin=10)
        img = gm.get_vardata()[..., :3]
        img[np.nonzero(gm.roi == 0)] /= 2.
        rmsd = np.sqrt(np.mean((ref - img)**2))
        self.assertTrue(rmsd < 0.2)

        gm = GoogleCenterMap(center_ll=(10.762660, 46.794221), zoom=13,
                             size_x=500, size_y=500)
        gm2 = GoogleCenterMap(center_ll=(10.762660, 46.794221), zoom=13,
                              size_x=500, size_y=500, scale=2)
        assert (gm.grid.nx * 2) == gm2.grid.nx
        assert gm.grid.extent == gm2.grid.extent

    @requires_internet
    @requires_motionless
    @requires_matplotlib
    def test_visible(self):
        import matplotlib as mpl

        x = [91.176036, 92.05, 88.880927]
        y = [29.649702, 31.483333, 29.264956]

        g = GoogleVisibleMap(x=x, y=y, size_x=400, size_y=400,
                             maptype='terrain')
        img = g.get_vardata()[..., :3]

        i, j = g.grid.transform(x, y, nearest=True)

        for _i, _j in zip(i, j):
            img[_j-3:_j+4, _i-3:_i+4, 0] = 1
            img[_j-3:_j+4, _i-3:_i+4, 1:] = 0

        # from PIL import Image
        # Image.fromarray((img * 255).astype(np.uint8)).save(
        #     get_demo_file('hef_google_visible.png'))
        ref = mpl.image.imread(get_demo_file('hef_google_visible.png'))
        rmsd = np.sqrt(np.mean((ref-img)**2))
        self.assertTrue(rmsd < 1e-1)

        self.assertRaises(ValueError, GoogleVisibleMap, x=x, y=y, zoom=12)

        fw = get_demo_file('wrf_tip_d1.nc')
        d = GeoNetcdf(fw)
        i, j = d.grid.ij_coordinates
        g = GoogleVisibleMap(x=i, y=j, crs=d.grid, size_x=500, size_y=500)
        img = g.get_vardata()[..., :3]
        mask = g.grid.map_gridded_data(i*0+1, d.grid)

        img[np.nonzero(mask)] = np.clip(img[np.nonzero(mask)] + 0.3, 0, 1)

        # from PIL import Image
        # Image.fromarray((img * 255).astype(np.uint8)).save(
        #     get_demo_file('hef_google_visible_grid.png'))
        ref = mpl.image.imread(get_demo_file('hef_google_visible_grid.png'))
        rmsd = np.sqrt(np.mean((ref-img)**2))
        self.assertTrue(rmsd < 5e-1)

        gm = GoogleVisibleMap(x=i, y=j, crs=d.grid,
                              size_x=500, size_y=500)
        gm2 = GoogleVisibleMap(x=i, y=j, crs=d.grid, scale=2,
                              size_x=500, size_y=500)
        assert (gm.grid.nx * 2) == gm2.grid.nx
        assert gm.grid.extent == gm2.grid.extent

        # Test regression for non array inputs
        grid = mercator_grid(center_ll=(72.5, 30.),
                             extent=(2.0e6, 2.0e6))
        GoogleVisibleMap(x=[0, grid.nx - 1], y=[0, grid.ny - 1], crs=grid)


class TestWRF(unittest.TestCase):

    def test_unstagger(self):

        wf = get_demo_file('wrf_cropped.nc')
        with netCDF4.Dataset(wf) as nc:
            nc.set_auto_mask(False)

            ref = nc['PH'][:]
            ref = 0.5 * (ref[:, :-1, ...] + ref[:, 1:, ...])

            # Own constructor
            v = wrftools.Unstaggerer(nc['PH'])
            assert_allclose(v[:], ref)
            assert_allclose(v[0:2, 2:12, ...],
                            ref[0:2, 2:12, ...])
            assert_allclose(v[:, 2:12, ...],
                            ref[:, 2:12, ...])
            assert_allclose(v[0:2, 2:12, 5:10, 15:17],
                            ref[0:2, 2:12, 5:10, 15:17])
            assert_allclose(v[1:2, 2:, 5:10, 15:17],
                            ref[1:2, 2:, 5:10, 15:17])
            assert_allclose(v[1:2, :-2, 5:10, 15:17],
                            ref[1:2, :-2, 5:10, 15:17])
            assert_allclose(v[1:2, 2:-4, 5:10, 15:17],
                            ref[1:2, 2:-4, 5:10, 15:17])
            assert_allclose(v[[0, 2], ...],
                            ref[[0, 2], ...])
            assert_allclose(v[..., [0, 2]],
                            ref[..., [0, 2]])
            assert_allclose(v[0, ...], ref[0, ...])

            # Under WRF
            nc = WRF(wf)
            assert_allclose(nc.get_vardata('PH'), ref)
            nc.set_period(1, 2)
            assert_allclose(nc.get_vardata('PH'), ref[1:3, ...])

    def test_unstagger_compressed(self):

        wf = get_demo_file('wrf_cropped.nc')
        wfc = get_demo_file('wrf_cropped_compressed.nc')

        # Under WRF
        nc = WRF(wf)
        ncc = WRF(wfc)
        assert_allclose(nc.get_vardata('PH'), ncc.get_vardata('PH'), rtol=.003)
        nc.set_period(1, 2)
        ncc.set_period(1, 2)
        assert_allclose(nc.get_vardata('PH'), ncc.get_vardata('PH'), rtol=.003)

    def test_ncl_diagvars(self):

        wf = get_demo_file('wrf_cropped.nc')
        ncl_out = get_demo_file('wrf_cropped_ncl.nc')

        w = WRF(wf)

        with netCDF4.Dataset(ncl_out) as nc:
            nc.set_auto_mask(False)

            ref = nc.variables['TK'][:]
            tot = w.get_vardata('TK')
            assert_allclose(ref, tot, rtol=1e-6)

            ref = nc.variables['SLP'][:]
            tot = w.get_vardata('SLP')
            assert_allclose(ref, tot, rtol=1e-6)

    def test_ncl_diagvars_compressed(self):

        wf = get_demo_file('wrf_cropped_compressed.nc')
        ncl_out = get_demo_file('wrf_cropped_ncl.nc')

        w = WRF(wf)

        with netCDF4.Dataset(ncl_out) as nc:
            nc.set_auto_mask(False)

            ref = nc.variables['TK'][:]
            tot = w.get_vardata('TK')
            assert_allclose(ref, tot, rtol=1e-5)

            ref = nc.variables['SLP'][:]
            tot = w.get_vardata('SLP')
            assert_allclose(ref, tot, rtol=1e-4)

    def test_staggeredcoords(self):

        wf = get_demo_file('wrf_cropped.nc')
        nc = GeoNetcdf(wf)
        lon, lat = nc.grid.xstagg_ll_coordinates
        assert_allclose(np.squeeze(nc.variables['XLONG_U'][0, ...]), lon,
                        atol=1e-4)
        assert_allclose(np.squeeze(nc.variables['XLAT_U'][0, ...]), lat,
                        atol=1e-4)
        lon, lat = nc.grid.ystagg_ll_coordinates
        assert_allclose(np.squeeze(nc.variables['XLONG_V'][0, ...]), lon,
                        atol=1e-4)
        assert_allclose(np.squeeze(nc.variables['XLAT_V'][0, ...]), lat,
                        atol=1e-4)

    def test_staggeredcoords_compressed(self):

        wf = get_demo_file('wrf_cropped_compressed.nc')
        nc = GeoNetcdf(wf)
        lon, lat = nc.grid.xstagg_ll_coordinates
        assert_allclose(np.squeeze(nc.variables['XLONG_U'][0, ...]), lon,
                        atol=1e-4)
        assert_allclose(np.squeeze(nc.variables['XLAT_U'][0, ...]), lat,
                        atol=1e-4)
        lon, lat = nc.grid.ystagg_ll_coordinates
        assert_allclose(np.squeeze(nc.variables['XLONG_V'][0, ...]), lon,
                        atol=1e-4)
        assert_allclose(np.squeeze(nc.variables['XLAT_V'][0, ...]), lat,
                        atol=1e-4)

    def test_har(self):

        # HAR
        hf = get_demo_file('har_d30km_y_2d_t2_2000.nc')
        d = GeoNetcdf(hf)
        reflon = np.squeeze(d.get_vardata('lon'))
        reflat = np.squeeze(d.get_vardata('lat'))
        mylon, mylat = d.grid.ll_coordinates
        np.testing.assert_allclose(reflon, mylon, atol=1e-5)
        np.testing.assert_allclose(reflat, mylat, atol=1e-5)