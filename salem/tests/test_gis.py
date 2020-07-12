from __future__ import division

import unittest
import warnings
import os

import pyproj
import pytest
import numpy as np
import netCDF4
from numpy.testing import assert_array_equal, assert_allclose

from salem import Grid
from salem import wgs84
import salem.gis as gis
from salem.utils import get_demo_file
from salem.tests import (requires_shapely, requires_geopandas,
                         requires_cartopy, requires_rasterio, python_version)


class SimpleNcDataSet():
    """Exploratory object to play around. For testing only."""

    def __init__(self, file):
        self.nc = netCDF4.Dataset(file)
        self.nc.set_auto_mask(False)
        proj = gis.check_crs(str(self.nc.proj4_str))
        x = self.nc.variables['x']
        y = self.nc.variables['y']
        dxdy = (x[1]-x[0], y[1]-y[0])
        nxny = (len(x), len(y))
        x0y0 = None
        if dxdy[1] > 0:
            x0y0 = (x[0], y[0])
        if dxdy[1] < 0:
            x0y0 = (x[0], y[0])
        self.grid = Grid(nxny=nxny, dxdy=dxdy, proj=proj, x0y0=x0y0)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self):
        self.nc.close()


class TestGrid(unittest.TestCase):

    def test_constructor(self):

        # It should work exact same for any projection
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:
            args = dict(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=proj)
            g = Grid(**args)
            self.assertTrue(isinstance(g, Grid))
            self.assertEqual(g.center_grid, g.corner_grid)

            # serialization
            d = g.to_dict()
            rg = Grid.from_dict(d)
            self.assertEqual(g, rg)
            g.to_json('test.json')
            rg = Grid.from_json('test.json')
            os.remove('test.json')
            self.assertEqual(g, rg)

            oargs = dict(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=proj)
            og = Grid(**oargs)
            self.assertEqual(g, og)

            # very simple test
            exp_i, exp_j = np.meshgrid(np.arange(3), np.arange(3))
            i, j = g.ij_coordinates
            assert_allclose(i, exp_i)
            assert_allclose(j, exp_j)

            i, j = g.xy_coordinates
            assert_allclose(i, exp_i)
            assert_allclose(j, exp_j)

            if proj == projs[0]:
                i, j = g.ll_coordinates
                assert_allclose(i, exp_i)
                assert_allclose(j, exp_j)

            args['proj'] = 'dummy'
            self.assertRaises(ValueError, Grid, **args)

            args['proj'] = proj
            args['nxny'] = (1, -1)
            self.assertRaises(ValueError, Grid, **args)
            args['nxny'] = (3, 3)

            args['dxdy'] = (1, -1)
            args['ll_corner'] = args['x0y0']
            del args['x0y0']
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.assertRaises(ValueError, Grid, **args)
                self.assertEqual(len(w), 1)
            args['x0y0'] = args['ll_corner']
            del args['ll_corner']

            # Center VS corner - multiple times because it was a bug
            assert_allclose(g.center_grid.xy_coordinates,
                                       g.xy_coordinates)
            assert_allclose(g.center_grid.center_grid.xy_coordinates,
                                       g.xy_coordinates)
            assert_allclose(g.corner_grid.corner_grid.xy_coordinates,
                                       g.corner_grid.xy_coordinates)

            ex = g.corner_grid.extent
            assert_allclose([-0.5,  2.5, -0.5,  2.5], ex)
            assert_allclose(g.center_grid.extent,
                                       g.corner_grid.extent)

            args['x0y0'] = (0, 0)

            g = Grid(**args)
            self.assertTrue(isinstance(g, Grid))

            oargs = dict(nxny=(3, 3), dxdy=(1, -1), x0y0=(0, 0), proj=proj)
            og = Grid(**oargs)
            self.assertEqual(g, og)

            # serialization
            d = og.to_dict()
            rg = Grid.from_dict(d)
            self.assertEqual(og, rg)

            # The simple test should work here too
            i, j = g.ij_coordinates
            assert_allclose(i, exp_i)
            assert_allclose(j, exp_j)

            # But the lonlats are the other way around:
            exp_x, exp_y = np.meshgrid(np.arange(3), -np.arange(3))
            x, y = g.xy_coordinates
            assert_allclose(x, exp_x)
            assert_allclose(y, exp_y)

            if proj == projs[0]:
                i, j = g.ll_coordinates
                assert_allclose(i, exp_x)
                assert_allclose(j, exp_y)

            # Center VS corner - multiple times because it was a bug
            assert_allclose(g.center_grid.xy_coordinates,
                                       g.xy_coordinates)
            assert_allclose(g.center_grid.center_grid.xy_coordinates,
                                       g.xy_coordinates)
            assert_allclose(g.corner_grid.corner_grid.xy_coordinates,
                                       g.corner_grid.xy_coordinates)

            ex = g.corner_grid.extent
            assert_allclose([-0.5,  2.5, -2.5,  0.5], ex)
            assert_allclose(g.center_grid.extent,
                                       g.corner_grid.extent)

            # The equivalents
            g = g.corner_grid
            i, j = g.ij_coordinates
            assert_allclose(i, exp_i)
            assert_allclose(j, exp_j)

            exp_x, exp_y = np.meshgrid(np.arange(3)-0.5, -np.arange(3)+0.5)
            x, y = g.xy_coordinates
            assert_allclose(x, exp_x)
            assert_allclose(y, exp_y)

            args = dict(nxny=(3, 2), dxdy=(1, 1), x0y0=(0, 0))
            g = Grid(**args)
            self.assertTrue(isinstance(g, Grid))
            self.assertTrue(g.xy_coordinates[0].shape == (2, 3))
            self.assertTrue(g.xy_coordinates[1].shape == (2, 3))

    def test_comparisons(self):
        """See if the grids can compare themselves"""

        args = dict(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        g1 = Grid(**args)
        self.assertEqual(g1.center_grid, g1.corner_grid)
        self.assertTrue(g1.center_grid.almost_equal(g1.center_grid))

        g2 = Grid(**args)
        self.assertEqual(g1, g2)
        self.assertTrue(g1.almost_equal(g2))

        args['dxdy'] = (1. + 1e-6, 1. + 1e-6)
        g2 = Grid(**args)
        self.assertNotEqual(g1, g2)
        self.assertTrue(g1.almost_equal(g2))

        # serialization
        d = g1.to_dict()
        rg = Grid.from_dict(d)
        self.assertEqual(g1, rg)
        d = g2.to_dict()
        rg = Grid.from_dict(d)
        self.assertEqual(g2, rg)
        self.assertNotEqual(g1, rg)
        self.assertTrue(g1.almost_equal(rg))
        g1.to_json('test.json')
        rg = Grid.from_json('test.json')
        os.remove('test.json')
        self.assertEqual(g1, rg)
        g2.to_json('test.json')
        rg = Grid.from_json('test.json')
        os.remove('test.json')
        self.assertEqual(g2, rg)
        self.assertNotEqual(g1, rg)
        self.assertTrue(g1.almost_equal(rg))

        args['proj'] = gis.check_crs('epsg:26915')
        g2 = Grid(**args)
        self.assertNotEqual(g1, g2)
        self.assertFalse(g1.almost_equal(g2))

        # New instance, same proj
        args['proj'] = gis.check_crs('epsg:26915')
        g1 = Grid(**args)
        self.assertEqual(g1, g2)
        self.assertTrue(g1.almost_equal(g2))
        # serialization
        d = g1.to_dict()
        rg = Grid.from_dict(d)
        self.assertEqual(g1, rg)
        self.assertTrue(g1.almost_equal(rg))
        g1.to_json('test.json')
        rg = Grid.from_json('test.json')
        os.remove('test.json')
        self.assertEqual(g1, rg)
        self.assertTrue(g1.almost_equal(rg))

    def test_reprs(self):
        from textwrap import dedent

        args = dict(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        g1 = Grid(**args)

        self.assertEqual(g1.__repr__(), g1.__str__())

    def test_errors(self):
        """Check that errors are occurring"""

        # It should work exact same for any projection
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                args = dict(nxny=(3, 3), dxdy=(1, -1), ll_corner=(0, 0), proj=proj)
                self.assertRaises(ValueError, Grid, **args)
                args = dict(nxny=(3, 3), dxdy=(-1, 0), ul_corner=(0, 0), proj=proj)
                self.assertRaises(ValueError, Grid, **args)
                args = dict(nxny=(3, 3), dxdy=(1, 1), proj=proj)
                self.assertRaises(ValueError, Grid, **args)
                args = dict(nxny=(3, -3), dxdy=(1, 1), ll_corner=(0, 0), proj=proj)
                self.assertRaises(ValueError, Grid, **args)
                args = dict(nxny=(3, 3), dxdy=(1, 1), ll_corner=(0, 0),
                            proj=proj, pixel_ref='areyoudumb')
                self.assertRaises(ValueError, Grid, **args)

                args = dict(nxny=(3, 3), dxdy=(1, 1), ll_corner=(0, 0), proj=proj)
                g = Grid(**args)
                self.assertRaises(ValueError, g.transform, 0, 0, crs=None)
                self.assertRaises(ValueError, g.transform, 0, 0, crs='areyou?')
                self.assertRaises(ValueError, g.map_gridded_data,
                                  np.zeros((3, 3)), 'areyou?')
                self.assertRaises(ValueError, g.map_gridded_data,
                                  np.zeros(3), g)
                self.assertRaises(ValueError, g.map_gridded_data,
                                  np.zeros((3, 4)), g)
                self.assertRaises(ValueError, g.map_gridded_data,
                                  np.zeros((3, 3)), g, interp='youare')

        # deprecation warnings
        for proj in projs:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                args = dict(nxny=(3, 3), dxdy=(1, -1), corner=(0, 0),
                            proj=proj)
                Grid(**args)
                args = dict(nxny=(3, 3), dxdy=(1, -1), ul_corner=(0, 0),
                            proj=proj)
                Grid(**args)
                args = dict(nxny=(3, 3), dxdy=(1, 1), ll_corner=(0, 0),
                            proj=proj)
                Grid(**args)
                if python_version == 'py3':
                    self.assertEqual(len(w), 3)

    def test_ij_to_crs(self):
        """Converting to projection"""

        # It should work exact same for any projection
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:

            args = dict(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=proj)

            g = Grid(**args)
            exp_i, exp_j = np.meshgrid(np.arange(3), np.arange(3))
            r_i, r_j = g.ij_to_crs(exp_i, exp_j)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            proj_out = proj
            r_i, r_j = g.ij_to_crs(exp_i, exp_j, crs=proj_out)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)

            # The equivalents
            gc = g.corner_grid
            r_i, r_j = gc.ij_to_crs(exp_i+0.5, exp_j+0.5)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            gc = g.center_grid
            r_i, r_j = gc.ij_to_crs(exp_i, exp_j)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)

            args = dict(nxny=(3, 3), dxdy=(1, -1), x0y0=(0, 0), proj=proj)
            g = Grid(**args)
            exp_i, exp_j = np.meshgrid(np.arange(3), -np.arange(3))
            in_i, in_j = np.meshgrid(np.arange(3), np.arange(3))
            r_i, r_j = g.ij_to_crs(in_i, in_j)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            proj_out = proj
            r_i, r_j = g.ij_to_crs(in_i, in_j, crs=proj_out)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)

            # The equivalents
            gc = g.corner_grid
            r_i, r_j = gc.ij_to_crs(in_i, in_j)
            assert_allclose(exp_i-0.5, r_i, atol=1e-03)
            assert_allclose(exp_j+0.5, r_j, atol=1e-03)
            gc = g.center_grid
            r_i, r_j = gc.ij_to_crs(in_i, in_j)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)

            # if we take some random projection it wont work
            proj_out = pyproj.Proj(proj="utm", zone=10, datum='NAD27')
            r_i, r_j = g.ij_to_crs(exp_i, exp_j, crs=proj_out)
            self.assertFalse(np.allclose(exp_i, r_i))
            self.assertFalse(np.allclose(exp_j, r_j))

            # Raise
            self.assertRaises(ValueError, g.ij_to_crs, exp_i, exp_j, crs='ups')

    def test_regrid(self):
        """New grids"""

        # It should work exact same for any projection
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:

            kargs = [dict(nxny=(3, 2), dxdy=(1, 1), x0y0=(0, 0),
                         proj=proj),
                     dict(nxny=(3, 2), dxdy=(1, -1), x0y0=(0, 0),
                         proj=proj),
                     dict(nxny=(3, 2), dxdy=(1, 1), x0y0=(0, 0),
                         proj=proj, pixel_ref='corner'),
                     dict(nxny=(3, 2), dxdy=(1, -1), x0y0=(0, 0),
                         proj=proj, pixel_ref='corner')]

            for ka in kargs:
                g = Grid(**ka)

                rg = g.regrid()
                self.assertTrue(g == rg)

                rg = g.regrid(factor=3)
                assert_array_equal(g.extent, rg.extent)
                assert_array_equal(g.extent, rg.extent)

                bg = rg.regrid(factor=1/3)
                self.assertEqual(g, bg)

                gx, gy = g.center_grid.xy_coordinates
                rgx, rgy = rg.center_grid.xy_coordinates
                assert_allclose(gx, rgx[1::3, 1::3], atol=1e-7)
                assert_allclose(gy, rgy[1::3, 1::3], atol=1e-7)

                gx, gy = g.center_grid.ll_coordinates
                rgx, rgy = rg.center_grid.ll_coordinates
                assert_allclose(gx, rgx[1::3, 1::3], atol=1e-7)
                assert_allclose(gy, rgy[1::3, 1::3], atol=1e-7)

                nrg = g.regrid(nx=9)
                self.assertTrue(nrg == rg)

                nrg = g.regrid(ny=6)
                self.assertTrue(nrg == rg)

    def test_transform(self):
        """Converting to the grid"""

        # It should work exact same for any projection
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:

            args = dict(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=proj)

            g = Grid(**args)
            exp_i, exp_j = np.meshgrid(np.arange(3), np.arange(3))
            r_i, r_j = g.transform(exp_i, exp_j, crs=proj)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            r_i, r_j = g.transform(exp_i, exp_j, crs=g)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            r_i, r_j = g.corner_grid.transform(exp_i, exp_j, crs=proj)
            assert_allclose(exp_i+0.5, r_i, atol=1e-03)
            assert_allclose(exp_j+0.5, r_j, atol=1e-03)
            r_i, r_j = g.corner_grid.transform(exp_i, exp_j, crs=g)
            assert_allclose(exp_i+0.5, r_i, atol=1e-03)
            assert_allclose(exp_j+0.5, r_j, atol=1e-03)

            args['pixel_ref'] = 'corner'
            g = Grid(**args)
            exp_i, exp_j = np.meshgrid(np.arange(3), np.arange(3))
            r_i, r_j = g.transform(exp_i, exp_j, crs=proj)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            r_i, r_j = g.transform(exp_i, exp_j, crs=g)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            r_i, r_j = g.corner_grid.transform(exp_i, exp_j, crs=proj)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            r_i, r_j = g.corner_grid.transform(exp_i, exp_j, crs=g)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            r_i, r_j = g.center_grid.transform(exp_i, exp_j, crs=proj)
            assert_allclose(exp_i-0.5, r_i, atol=1e-03)
            assert_allclose(exp_j-0.5, r_j, atol=1e-03)
            r_i, r_j = g.center_grid.transform(exp_i, exp_j, crs=g)
            assert_allclose(exp_i-0.5, r_i, atol=1e-03)
            assert_allclose(exp_j-0.5, r_j, atol=1e-03)
            ex = g.corner_grid.extent
            assert_allclose([0, 3, 0, 3], ex, atol=1e-03)
            assert_allclose(g.center_grid.extent,
                                       g.corner_grid.extent,
                                       atol=1e-03)


            # Masked
            xi = [-0.6, 0.5, 1.2, 2.9, 3.1, 3.6]
            yi = xi
            ex = [-1, 0, 1, 2, 3, 3]
            ey = ex
            r_i, r_j = g.corner_grid.transform(xi, yi, crs=proj)
            assert_allclose(xi, r_i, atol=1e-03)
            assert_allclose(yi, r_j, atol=1e-03)
            r_i, r_j = g.corner_grid.transform(xi, yi, crs=proj, nearest=True)
            assert_array_equal(ex, r_i)
            assert_array_equal(ey, r_j)
            r_i, r_j = g.center_grid.transform(xi, yi, crs=proj, nearest=True)
            assert_array_equal(ex, r_i)
            assert_array_equal(ey, r_j)
            ex = np.ma.masked_array(ex, mask=[1, 0, 0, 0, 1, 1])
            ey = ex
            r_i, r_j = g.center_grid.transform(xi, yi, crs=proj,
                                               nearest=True, maskout=True)
            assert_array_equal(ex, r_i)
            assert_array_equal(ey, r_j)
            assert_array_equal(ex.mask, r_i.mask)
            assert_array_equal(ey.mask, r_j.mask)
            r_i, r_j = g.corner_grid.transform(xi, yi, crs=proj,
                                               nearest=True, maskout=True)
            assert_array_equal(ex, r_i)
            assert_array_equal(ey, r_j)
            assert_array_equal(ex.mask, r_i.mask)
            assert_array_equal(ey.mask, r_j.mask)

            del args['pixel_ref']
            args['x0y0'] = (0, 0)
            args['dxdy'] = (1, -1)
            g = Grid(**args)
            in_i, in_j = np.meshgrid(np.arange(3), -np.arange(3))
            exp_i, exp_j = np.meshgrid(np.arange(3), np.arange(3))
            r_i, r_j = g.transform(in_i, in_j, crs=proj)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            in_i, in_j = np.meshgrid(np.arange(3), np.arange(3))
            r_i, r_j = g.transform(in_i, in_j, crs=g)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)

    def test_lookup_grid(self):

        data = np.arange(12).reshape((4, 3))
        args = dict(nxny=(3, 4), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        g = Grid(**args)
        lut = g.grid_lookup(g)
        for ji, l in lut.items():
            self.assertEqual(data[ji], data[l[:, 0], l[:, 1]])

        args = dict(nxny=(2, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        g2 = Grid(**args)
        lut = g2.grid_lookup(g)
        for ji, l in lut.items():
            self.assertEqual(data[ji], data[l[:, 0], l[:, 1]])

        lut = g.grid_lookup(g2)
        for (j, i), l in lut.items():
            if j > 2 or i > 1:
                assert l is None
            else:
                self.assertEqual(data[j, i], data[l[:, 0], l[:, 1]])

        args = dict(nxny=(1, 1), dxdy=(10, 10), x0y0=(0, 0), proj=wgs84)
        g3 = Grid(**args)

        lut = g3.grid_lookup(g)
        od = data[lut[(0, 0)][:, 0], lut[(0, 0)][:, 1]]
        self.assertEqual(len(od), 12)
        assert_allclose(np.mean(od), np.mean(data))

    def test_lookup_transform(self):

        data2d = np.arange(12).reshape((4, 3))
        data3d = np.stack([data2d, data2d, data2d])
        data4d = np.stack([data3d, data3d])

        args = dict(nxny=(3, 4), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        g = Grid(**args)

        odata = g.lookup_transform(data2d, g)
        assert_allclose(odata, data2d)
        odata = g.lookup_transform(data3d, g)
        assert_allclose(odata, data3d)
        odata = g.lookup_transform(data4d, g)
        assert_allclose(odata, data4d)
        odata, lut = g.lookup_transform(data4d, g, method=len, return_lut=True)
        assert_allclose(odata, data4d*0+1)

        # set lut
        odata = g.lookup_transform(data2d, g, lut=lut)
        assert_allclose(odata, data2d)

        args = dict(nxny=(2, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        g2 = Grid(**args)
        odata = g2.lookup_transform(data2d, g)
        assert_allclose(odata, data2d[:-1, :-1])
        odata = g2.lookup_transform(data3d, g)
        assert_allclose(odata, data3d[..., :-1, :-1])
        odata = g2.lookup_transform(data4d, g)
        assert_allclose(odata, data4d[..., :-1, :-1])

        f = self.assertRaisesRegex if python_version == 'py3' else self.assertRaisesRegexp
        with f(ValueError, 'dimension not compatible'):
            g.lookup_transform(data2d[:-1, :-1], g)

        odata = g.lookup_transform(data2d[:-1, :-1], g2)
        ref = data2d * 0
        ref[-1, :] = 1
        ref[:, -1] = 1
        assert_allclose(odata.mask, ref)
        assert_allclose(data2d, odata)

        odata = g.lookup_transform(data2d[:-1, :-1], g2, method=len)
        assert_allclose(odata, 1-ref)

        args = dict(nxny=(1, 1), dxdy=(10, 10), x0y0=(0, 0), proj=wgs84)
        g3 = Grid(**args)

        odata = g3.lookup_transform(data2d, g)
        self.assertEqual(odata.shape, (1, 1))
        assert_allclose(odata, np.mean(data2d))

        odata = g3.lookup_transform(data2d, g, method=np.sum)
        self.assertEqual(odata.shape, (1, 1))
        assert_allclose(odata, np.sum(data2d))

        odata = g3.lookup_transform(data2d, g, method=len)
        self.assertEqual(odata.shape, (1, 1))
        assert_allclose(odata, 12)

        # total back and forth
        data = np.arange(12).reshape((4, 3))
        args = dict(nxny=(3, 4), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        g = Grid(**args)
        rg = g.regrid(factor=3)
        tdata = rg.map_gridded_data(data, g, interp='nearest')
        odata = g.lookup_transform(tdata, rg)
        assert_allclose(odata, data)
        odata = g.lookup_transform(tdata, rg, method=len)
        assert_allclose(odata, data*0.+9)

    def test_stagg(self):
        """Staggered grids."""

        # It should work exact same for any projection
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:
            args = dict(nxny=(3, 2), dxdy=(1, 1), x0y0=(0, 0),
                        proj=proj, pixel_ref='corner')
            g = Grid(**args)
            x, y = g.xstagg_xy_coordinates
            assert_array_equal(x, np.array([[0,1,2,3], [0,1,2,3]]))
            assert_array_equal(y, np.array([[0.5,  0.5,  0.5,  0.5],
                                            [1.5,  1.5,  1.5,  1.5]]))
            xx, yy = g.corner_grid.xstagg_xy_coordinates
            assert_array_equal(x, xx)
            assert_array_equal(y, yy)
            xt, yt = x, y

            x, y = g.ystagg_xy_coordinates
            assert_array_equal(x, np.array([[0.5,  1.5,  2.5],
                                            [0.5,  1.5,  2.5],
                                            [0.5,  1.5,  2.5]]))
            assert_array_equal(y, np.array([[0,  0,  0],
                                            [1,  1,  1],
                                            [2,  2,  2]]))
            xx, yy = g.corner_grid.ystagg_xy_coordinates
            assert_array_equal(x, xx)
            assert_array_equal(y, yy)

            if proj is wgs84:
                xx, yy = g.corner_grid.ystagg_ll_coordinates
                assert_allclose(x, xx)
                assert_allclose(y, yy)
                xx, yy = g.corner_grid.xstagg_ll_coordinates
                assert_allclose(xt, xx)
                assert_allclose(yt, yy)

                x, y = g.pixcorner_ll_coordinates
                assert_allclose(x, np.array([[0, 1, 2, 3],
                                             [0, 1, 2, 3],
                                             [0, 1, 2, 3]]))
                assert_allclose(y, np.array([[0, 0, 0, 0],
                                             [1, 1, 1, 1],
                                             [2, 2, 2, 2]]))

    def test_map_gridded_data(self):
        """Ok now the serious stuff starts with some fake data"""

        # It should work exact same for any projection
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:
            nx, ny = (3, 4)
            data = np.arange(nx * ny).reshape((ny, nx))

            # Nearest Neighbor
            args = dict(nxny=(nx, ny), dxdy=(1, 1), x0y0=(0, 0), proj=proj)
            g = Grid(**args)
            odata = g.map_gridded_data(data, g)
            self.assertTrue(odata.shape == data.shape)
            assert_allclose(data, odata, atol=1e-03)

            # Out of the grid
            go = Grid(nxny=(nx, ny), dxdy=(1, 1), x0y0=(9, 9), proj=proj)
            odata = g.map_gridded_data(data, go)
            odata.set_fill_value(-999)
            self.assertTrue(odata.shape == data.shape)
            self.assertTrue(np.all(odata.mask))

            args = dict(nxny=(nx - 1, ny - 1), dxdy=(1, 1), x0y0=(0, 0),
                        proj=proj)
            ig = Grid(**args)
            odata = g.map_gridded_data(data[0:ny - 1, 0:nx - 1], ig)
            self.assertTrue(odata.shape == (ny, nx))
            assert_allclose(data[0:ny - 1, 0:nx - 1],
                            odata[0:ny - 1, 0:nx - 1], atol=1e-03)
            assert_array_equal([True] * 3, odata.mask[ny - 1, :])

            data = np.arange(nx * ny).reshape((ny, nx)) * 1.2
            odata = g.map_gridded_data(data[0:ny - 1, 0:nx - 1], ig)
            self.assertTrue(odata.shape == (ny, nx))
            assert_allclose(data[0:ny - 1, 0:nx - 1],
                            odata[0:ny - 1, 0:nx - 1], atol=1e-03)
            self.assertTrue(
                np.sum(np.isfinite(odata)) == ((ny - 1) * (nx - 1)))

            # Bilinear
            data = np.arange(nx * ny).reshape((ny, nx))
            exp_data = np.array([2., 3., 5., 6., 8., 9.]).reshape(
                (ny - 1, nx - 1))
            args = dict(nxny=(nx, ny), dxdy=(1, 1), x0y0=(0, 0), proj=proj)
            gfrom = Grid(**args)
            args = dict(nxny=(nx - 1, ny - 1), dxdy=(1, 1), x0y0=(0.5, 0.5),
                        proj=proj)
            gto = Grid(**args)
            odata = gto.map_gridded_data(data, gfrom, interp='linear')
            self.assertTrue(odata.shape == (ny - 1, nx - 1))
            assert_allclose(exp_data, odata, atol=1e-03)

    def test_map_gridded_data_over(self):

        # It should work exact same for any projection
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:
            nx, ny = (4, 5)
            data = np.arange(nx * ny).reshape((ny, nx)).astype(np.float)

            in_data = data * np.NaN
            in_data[0, :] = 78

            # Nearest Neighbor
            args = dict(nxny=(nx, ny), dxdy=(1, 1), x0y0=(0, 0), proj=proj)
            g = Grid(**args)
            odata = g.map_gridded_data(data, g, out=data.copy())
            self.assertTrue(odata.shape == data.shape)
            assert_allclose(data, odata, atol=1e-03)

            odata = g.map_gridded_data(in_data, g, out=data.copy())
            self.assertTrue(odata.shape == data.shape)
            assert_allclose(data[1:, :], odata[1:, :], atol=1e-03)
            assert_allclose(odata[0, :], 78, atol=1e-03)

            # Bilinear
            odata = g.map_gridded_data(data, g, interp='linear',
                                       out=data.copy())
            self.assertTrue(odata.shape == data.shape)
            assert_allclose(data, odata, atol=1e-03)

            # Spline
            odata = g.map_gridded_data(data, g, interp='spline',
                                       out=data.copy())
            self.assertTrue(odata.shape == data.shape)
            assert_allclose(data, odata, atol=1e-03)


    @requires_shapely
    def test_extent(self):

        # It should work exact same for any projection
        args = dict(nxny=(9, 9), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84)
        g1 = Grid(**args)
        assert_allclose(g1.extent, g1.extent_in_crs(crs=g1.proj), atol=1e-3)

        args = dict(nxny=(9, 9), dxdy=(30000, 30000), x0y0=(0., 1577463),
                    proj=gis.check_crs('epsg:26915'))
        g2 = Grid(**args)
        assert_allclose(g2.extent, g2.extent_in_crs(crs=g2.proj), atol=1e-3)

        exg = np.array(g2.extent_in_crs(crs=g1))
        exgx, exgy = g1.ij_to_crs(exg[[0, 1]], exg[[2, 3]], crs=wgs84)

        lon, lat = g2.corner_grid.ll_coordinates
        assert_allclose([np.min(lon), np.min(lat)], [exgx[0], exgy[0]],
                        rtol=0.1)

        p = g2.extent_as_polygon(crs=g2.proj)

        assert p.is_valid
        x, y = p.exterior.coords.xy
        assert_allclose([np.min(x), np.max(x), np.min(y), np.max(y)],
                        g2.extent)

    def test_simple_dataset(self):
        # see if with is working
        with SimpleNcDataSet(get_demo_file('dem_wgs84.nc')) as nc:
            nc = SimpleNcDataSet(get_demo_file('dem_wgs84.nc'))
            grid_from = nc.grid
        self.assertTrue(gis.check_crs(grid_from))

    def test_map_real_data(self):
        """Ok now the serious stuff starts with some real data"""

        nc = SimpleNcDataSet(get_demo_file('dem_wgs84.nc'))
        data_from = nc.nc.variables['dem'][:]
        grid_from = nc.grid

        # DL corner
        nc = SimpleNcDataSet(get_demo_file('dem_mercator.nc'))
        data_gdal = nc.nc.variables['dem_gdal'][:]
        grid_to = nc.grid
        odata = grid_to.map_gridded_data(data_from, grid_from, interp='linear')
        assert_allclose(data_gdal, odata)

        # UL corner, no change needed to the code
        nc = SimpleNcDataSet(get_demo_file('dem_mercator_ul.nc'))
        data_gdal = nc.nc.variables['dem_gdal']
        grid_to = nc.grid
        self.assertTrue(grid_to.origin == 'upper-left')
        odata = grid_to.map_gridded_data(data_from, grid_from, interp='linear')
        assert_allclose(data_gdal, odata)

        # Now Larger grids
        ncw = SimpleNcDataSet(get_demo_file('wrf_grid.nc'))
        nct = SimpleNcDataSet(get_demo_file('trmm_grid.nc'))

        # TRMM to WRF
        data = nct.nc.variables['prcp'][:]
        grid_from = nct.grid
        grid_to = ncw.grid

        ref_data = ncw.nc.variables['trmm_on_wrf_nn'][:]
        odata = grid_to.map_gridded_data(data, grid_from)
        assert_allclose(ref_data, odata)

        ref_data = ncw.nc.variables['trmm_on_wrf_bili'][:]
        odata = grid_to.map_gridded_data(data, grid_from, interp='linear')
        assert_allclose(ref_data, odata, rtol=1e-5)

        ref_data = ncw.nc.variables['trmm_on_wrf_bili'][:]
        odata = grid_to.map_gridded_data(data, grid_from, interp='spline')
        assert_allclose(ref_data, odata, atol=1e-1)

        # WRF to TRMM
        grid_from = ncw.grid
        grid_to = nct.grid

        # 3D
        data = ncw.nc.variables['wrf_t2'][:]
        with np.errstate(invalid='ignore'):
            # I think this is a NetCDF4 warning
            ref_data = nct.nc.variables['t2_on_trmm_bili'][:]
        odata = grid_to.map_gridded_data(data, grid_from, interp='linear')
        # At the borders IDL and Python take other decision on wether it
        # should be a NaN or not (Python seems to be more conservative)
        ref_data[np.where(odata.mask)] = np.NaN
        assert np.sum(np.isfinite(ref_data)) != 0
        assert_allclose(ref_data, odata.filled(np.NaN), atol=1e-3)

        odata = grid_to.map_gridded_data(data, grid_from, interp='spline')
        odata[np.where(~ np.isfinite(ref_data))] = np.NaN
        ref_data[np.where(~ np.isfinite(odata))] = np.NaN
        assert np.sum(np.isfinite(ref_data)) != 0
        assert_allclose(ref_data, odata, rtol=0.2, atol=3)

        # 4D
        data = np.array([data, data])
        ref_data = np.array([ref_data, ref_data])
        odata = grid_to.map_gridded_data(data, grid_from, interp='linear')
        odata = odata.filled(np.NaN)
        ref_data[np.where(~ np.isfinite(odata))] = np.NaN
        assert np.sum(np.isfinite(ref_data)) != 0
        assert_allclose(ref_data, odata, atol=1e-3)

        odata = grid_to.map_gridded_data(data, grid_from, interp='spline')
        odata[np.where(~ np.isfinite(ref_data))] = np.NaN
        ref_data[np.where(~ np.isfinite(odata))] = np.NaN
        assert np.sum(np.isfinite(ref_data)) != 0
        assert_allclose(ref_data, odata, rtol=0.2, atol=3)

        # 4D - INTEGER
        data = ncw.nc.variables['wrf_tk'][:]
        ref_data = nct.nc.variables['tk_on_trmm_nn'][:]
        odata = grid_to.map_gridded_data(data, grid_from)
        # At the borders IDL and Python take other decision on wether it
        # should be a NaN or not (Python seems to be more conservative)
        self.assertTrue(odata.dtype == ref_data.dtype)
        ref_data[np.where(odata == -999)] = -999
        assert_allclose(ref_data, odata.filled(-999))

    @requires_shapely
    def test_roi(self):

        import shapely.geometry as shpg

        g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84,
                 pixel_ref='corner')
        p = shpg.Polygon([(1.5, 1.), (2., 1.5), (1.5, 2.), (1., 1.5)])
        roi = g.region_of_interest(geometry=p)
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,0]], roi)

        roi = g.region_of_interest(corners=([0, 0], [2, 2]), crs=wgs84)
        np.testing.assert_array_equal([[1, 1, 1], [1, 1, 1], [1, 1, 1]], roi)

        roi = g.region_of_interest(corners=([1.3, 1.3], [1.7, 1.7]), crs=wgs84)
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,0]], roi)

        roi = g.region_of_interest()
        np.testing.assert_array_equal([[0,0,0],[0,0,0],[0,0,0]], roi)

        mask = [[0,0,0],[0,1,0],[0,0,0]]
        roi = g.region_of_interest(roi=mask)
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,0]], roi)

        nc = np.array(p.exterior.coords) + 0.1
        p = shpg.Polygon(nc)
        roi = g.region_of_interest(geometry=p, roi=roi)
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,0]], roi)

        nc = np.array(p.exterior.coords) + 0.5
        p = shpg.Polygon(nc)
        roi = g.region_of_interest(geometry=p, roi=roi)
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,0]], roi)

        nc = np.array(p.exterior.coords) + 0.5
        p = shpg.Polygon(nc)
        roi = g.region_of_interest(geometry=p, roi=roi)
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,1]], roi)

        g = Grid(nxny=(4, 2), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84,
                 pixel_ref='corner')
        p = shpg.Polygon([(1.5, 1.), (2., 1.5), (1.5, 2.), (1., 1.5)])
        roi = g.region_of_interest(geometry=p)
        np.testing.assert_array_equal([[0,0,0,0],[0,1,0,0]], roi)

        g = Grid(nxny=(2, 4), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84,
                 pixel_ref='corner')
        p = shpg.Polygon([(1.5, 1.), (2., 1.5), (1.5, 2.), (1., 1.5)])
        roi = g.region_of_interest(geometry=p)
        np.testing.assert_array_equal([[0,0], [0,1], [0,0], [0,0]], roi)

        g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84,
                 pixel_ref='corner')
        g2 = Grid(nxny=(1, 1), dxdy=(0.2, 0.2), x0y0=(1.4, 1.4),
                  proj=wgs84, pixel_ref='corner')
        roi = g.region_of_interest(grid=g2)
        np.testing.assert_array_equal([[0,0,0],[0,1,0],[0,0,0]], roi)

    def test_to_dataset(self):
        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:
            g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=proj)
            ds = g.to_dataset()
            self.assertTrue(g == ds.salem.grid)

            g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=proj,
                     pixel_ref='corner')
            ds = g.to_dataset()
            self.assertTrue(g == ds.salem.grid)

    @requires_geopandas
    def test_geometry(self):
        projs = [wgs84, gis.check_crs('epsg:26915')]
        from shapely.geometry import Point
        for proj in projs:
            g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0.5, 0.5), proj=proj)
            gdf = g.to_geometry()
            self.assertEqual(len(gdf), 9)
            self.assertTrue(gdf.contains(Point(1.5, 1.5))[4])
            self.assertFalse(gdf.contains(Point(1.5, 1.5))[5])

        gdf = g.to_geometry(to_crs=wgs84)
        # This is now quite off
        self.assertFalse(gdf.contains(Point(1.5, 1.5))[4])

    def test_xarray_support(self):
        # what happens if we use salem's funcs with xarray?
        import xarray as xr

        projs = [wgs84, gis.check_crs('epsg:26915')]

        for proj in projs:
            args = dict(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=proj)
            g = Grid(**args)
            exp_i, exp_j = np.meshgrid(np.arange(3), np.arange(3))
            exp_i, exp_j = (xr.DataArray(exp_i, dims=['y', 'x']),
                            xr.DataArray(exp_j, dims=['y', 'x']))
            r_i, r_j = g.ij_to_crs(exp_i, exp_j)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            self.assertTrue(r_i.dims == exp_i.dims)

            # transform
            r_i, r_j = g.transform(exp_i, exp_j, crs=proj)
            assert_allclose(exp_i, r_i, atol=1e-03)
            assert_allclose(exp_j, r_j, atol=1e-03)
            # TODO: this doesn't work:
            # self.assertTrue(r_i.dims == exp_i.dims)

            # map
            nx, ny = (3, 4)
            data = np.arange(nx * ny).reshape((ny, nx))
            data = xr.DataArray(data, coords={'y':np.arange(ny),
                                              'x':np.arange(nx)},
                                dims=['y', 'x'])
            data.attrs = {'test': 'attr'}

            # Nearest Neighbor
            args = dict(nxny=(nx, ny), dxdy=(1, 1), x0y0=(0, 0), proj=proj)
            g = Grid(**args)
            odata = g.map_gridded_data(data, g)
            self.assertTrue(odata.shape == data.shape)
            assert_allclose(data, odata, atol=1e-03)

            # Transform can understand a grid
            data.attrs['pyproj_srs'] = g.proj.srs
            odata = g.map_gridded_data(data)
            self.assertTrue(odata.shape == data.shape)
            assert_allclose(data, odata, atol=1e-03)


class TestTransform(unittest.TestCase):

    def test_check_crs_log(self):

        assert gis.check_crs('wrong') is None
        with pytest.raises(ValueError):
            gis.check_crs('wrong', raise_on_error=True)

    def test_same_proj(self):

        # this should work regardless of gdal or not:
        p1 = pyproj.Proj('+proj=utm +zone=15 +datum=NAD83 '
                         '+ellps=GRS80 +towgs84=0,0,0 +units=m +no_defs')
        p2 = pyproj.Proj('+proj=utm +zone=15 +datum=NAD83 +units=m +no_defs '
                         '+ellps=GRS80 +towgs84=0,0,0')
        self.assertTrue(gis.proj_is_same(p1, p2))

        # this needs gdal
        p1 = gis.check_crs('epsg:26915')
        p2 = pyproj.Proj('+proj=utm +zone=15 +ellps=GRS80 +datum=NAD83 '
                         '+units=m +no_defs')
        if gis.has_gdal:
            self.assertTrue(gis.proj_is_same(p1, p2))

    def test_pyproj_trafo(self):

        x = np.random.randn(int(1e6)) * 60
        y = np.random.randn(int(1e6)) * 60

        for i in np.arange(3):
            xx, yy = gis.transform_proj(wgs84, wgs84, x, y)
        assert_allclose(xx, x)
        assert_allclose(yy, y)

        for i in np.arange(3):
            xx, yy = gis.transform_proj(wgs84, wgs84, x, y, nocopy=True)
        assert_allclose(xx, x)
        assert_allclose(yy, y)

        xx, yy = gis.transform_proj(gis.check_crs('epsg:26915'),
                                    gis.check_crs('epsg:26915'), x, y)
        assert_allclose(xx, x)
        assert_allclose(yy, y)

    @requires_shapely
    def test_geometry(self):

        import shapely.geometry as shpg

        g = Grid(nxny=(3, 3), dxdy=(1, 1), x0y0=(0, 0), proj=wgs84,
                 pixel_ref='corner')
        p = shpg.Polygon([(1.5, 1.), (2., 1.5), (1.5, 2.), (1., 1.5)])
        o = gis.transform_geometry(p, to_crs=g)
        assert_allclose(p.exterior.coords, o.exterior.coords)

        q = gis.transform_geometry(o, crs=g)
        assert_allclose(p.exterior.coords, q.exterior.coords)

        o = gis.transform_geometry(p, to_crs=g.center_grid)
        totest = np.array(o.exterior.coords) + 0.5
        assert_allclose(p.exterior.coords, totest)

        x, y = g.corner_grid.xy_coordinates
        p = shpg.MultiPoint([shpg.Point(i, j) for i, j in zip(x.flatten(),
                                                              y.flatten())])
        o = gis.transform_geometry(p, to_crs=g.proj)
        assert_allclose([_p.coords for _p in o], [_p.coords for _p in p])

    @requires_geopandas
    def test_shape(self):
        """Is the transformation doing well?"""

        from salem import read_shapefile

        so = read_shapefile(get_demo_file('Hintereisferner.shp'))
        sref = read_shapefile(get_demo_file('Hintereisferner_UTM.shp'))
        st = gis.transform_geopandas(so, to_crs=sref.crs)
        self.assertFalse(st is so)
        assert_allclose(st.geometry[0].exterior.coords,
                                   sref.geometry[0].exterior.coords)

        sti = gis.transform_geopandas(so, to_crs=sref.crs, inplace=True)
        self.assertTrue(sti is so)
        assert_allclose(so.geometry[0].exterior.coords,
                                   sref.geometry[0].exterior.coords)
        assert_allclose(sti.geometry[0].exterior.coords,
                                   sref.geometry[0].exterior.coords)

        g = Grid(nxny=(1, 1), dxdy=(1, 1), x0y0=(10., 46.), proj=wgs84)
        so = read_shapefile(get_demo_file('Hintereisferner.shp'))
        st = gis.transform_geopandas(so, to_crs=g)

        ref = np.array(so.geometry[0].exterior.coords)
        ref = ref - np.floor(ref)
        assert_allclose(ref, st.geometry[0].exterior.coords)

        # round trip
        so_back = gis.transform_geopandas(st, from_crs=g, to_crs=so.crs)
        assert_allclose(so_back.geometry[0].exterior.coords,
                        so.geometry[0].exterior.coords)


class TestGrids(unittest.TestCase):

    def test_mercatorgrid(self):

        grid = gis.mercator_grid(center_ll=(11.38, 47.26),
                                         extent=(2000000, 2000000))
        lon1, lat1 = grid.center_grid.ll_coordinates
        e1 = grid.extent
        grid = gis.mercator_grid(center_ll=(11.38, 47.26),
                                 extent=(2000000, 2000000),
                                 origin='upper-left')
        lon2, lat2 = grid.center_grid.ll_coordinates
        e2 = grid.extent

        assert_allclose(e1, e2)
        assert_allclose(lon1, lon2[::-1, :])
        assert_allclose(lat1, lat2[::-1, :])

        grid = gis.mercator_grid(center_ll=(11.38, 47.26),
                                         extent=(2000, 2000),
                                         nx=100)
        lon1, lat1 = grid.pixcorner_ll_coordinates
        e1 = grid.extent
        grid = gis.mercator_grid(center_ll=(11.38, 47.26),
                                 extent=(2000, 2000),
                                 origin='upper-left',
                                 nx=100)
        lon2, lat2 = grid.pixcorner_ll_coordinates
        e2 = grid.extent

        assert_allclose(e1, e2)
        assert_allclose(lon1, lon2[::-1, :])
        assert_allclose(lat1, lat2[::-1, :])

        grid = gis.mercator_grid(center_ll=(11.38, 47.26),
                                         extent=(2000, 2000),
                                         nx=10)
        e1 = grid.extent
        grid = gis.mercator_grid(center_ll=(11.38, 47.26),
                                 extent=(2000, 2000),
                                 origin='upper-left',
                                 nx=9)
        e2 = grid.extent
        assert_allclose(e1, e2)


def fuzzy_proj_tester(p1, p2, atol=1e-16):

    d1 = dict()
    d2 = dict()
    for d, p in zip((d1, d2), (p1, p2)):
        for s in p.srs.split('+'):
            s = s.split('=')
            if len(s) != 2:
                continue
            k = s[0].strip()
            v = s[1].strip()
            try:
                v = float(v)
            except:
                pass
            d[k] = v

    for k in d1.keys():
        if k in d2:
            if d1[k] == d2[k]:
                # strings
                continue
            else:
                try:
                    assert_allclose(d1[k], d2[k], atol=atol,
                                    err_msg='key: {}'.format(k))
                except TypeError:
                    assert d1[k] == d2[k]


class TestCartopy(unittest.TestCase):

    @requires_cartopy
    @requires_rasterio
    def test_to_cartopy(self):

        import cartopy.crs as ccrs
        from salem import GeoNetcdf, GeoTiff

        grid = gis.mercator_grid(center_ll=(11.38, 47.26),
                                 extent=(2000000, 2000000))
        p = gis.proj_to_cartopy(grid.proj)
        assert isinstance(p, ccrs.TransverseMercator)
        fuzzy_proj_tester(grid.proj, pyproj.Proj(p.proj4_params))

        ds = GeoNetcdf(get_demo_file('wrfout_d01.nc'))
        p = gis.proj_to_cartopy(ds.grid.proj)
        assert isinstance(p, ccrs.LambertConformal)
        fuzzy_proj_tester(ds.grid.proj, pyproj.Proj(p.proj4_params))

        ds = GeoNetcdf(get_demo_file('wrf_mercator.nc'))
        p = gis.proj_to_cartopy(ds.grid.proj)
        assert isinstance(p, ccrs.Mercator)
        fuzzy_proj_tester(ds.grid.proj, pyproj.Proj(p.proj4_params))

        ds = GeoTiff(get_demo_file('himalaya.tif'))
        p = gis.proj_to_cartopy(ds.grid.proj)
        assert isinstance(p, ccrs.PlateCarree)

        ds = GeoTiff(get_demo_file('hef_roi.tif'))
        p = gis.proj_to_cartopy(ds.grid.proj)
        assert isinstance(p, ccrs.PlateCarree)

        p = gis.proj_to_cartopy(wgs84)
        assert isinstance(p, ccrs.PlateCarree)

        p = gis.proj_to_cartopy(pyproj.Proj('+proj=utm +zone=15'))
        assert isinstance(p, ccrs.UTM)

        # this needs gdal
        if gis.has_gdal:
            p = gis.proj_to_cartopy(gis.check_crs('epsg:26915'))
            assert isinstance(p, ccrs.UTM)
