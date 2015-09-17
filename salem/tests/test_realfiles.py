"""Tests on real world files. These files are large and contained in
an online directory:

Make the files available in order to run these tests.

Be sure ncl is installed on your sysem
"""
from __future__ import division

import unittest
import os

import netCDF4
import numpy as np
from numpy.testing import assert_allclose

from salem import datasets
from salem import wrf
from salem.utils import get_demo_file

myenv = os.environ.copy()

class TestDataset(unittest.TestCase):

    def test_unstagger(self):

        wf = get_demo_file('wrf_cropped.nc')
        nc = netCDF4.Dataset(wf)

        ref = nc['PH'][:]
        ref = 0.5 * (ref[:, :-1, ...] + ref[:, 1:, ...])

        # Own constructor
        v = wrf.Unstaggerer(nc['PH'])
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
        # TODO: this is an issue
        assert_allclose(v[0, ...], ref[0:0, ...])

        # Under WRF
        nc = datasets.WRF(wf)
        assert_allclose(nc.get_vardata('PH'), ref)
        nc.set_period(1, 2)
        assert_allclose(nc.get_vardata('PH'), ref[1:3, ...])


    def test_ncl_diagvars(self):

        wf = get_demo_file('wrf_cropped.nc')
        ncl_out = get_demo_file('wrf_cropped_ncl.nc')

        w = datasets.WRF(wf)

        nc = netCDF4.Dataset(ncl_out)
        ref = nc.variables['TK'][:]
        tot = w.get_vardata('TK')
        assert_allclose(ref, tot, rtol=1e-6)

        ref = nc.variables['SLP'][:]
        tot = w.get_vardata('SLP')
        assert_allclose(ref, tot, rtol=1e-6)

    def test_staggeredcoords(self):

        wf = get_demo_file('wrf_cropped.nc')
        nc = datasets.GeoNetcdf(wf)
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
        d = datasets.GeoNetcdf(hf)
        reflon = np.squeeze(d.get_vardata('lon'))
        reflat = np.squeeze(d.get_vardata('lat'))
        mylon, mylat = d.grid.ll_coordinates
        np.testing.assert_allclose(reflon, mylon, atol=1e-5)
        np.testing.assert_allclose(reflat, mylat, atol=1e-5)