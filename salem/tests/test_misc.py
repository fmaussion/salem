from __future__ import annotations
import copy
import shutil
import time
import unittest
from pathlib import Path

import netCDF4
import numpy as np
import pytest
from numpy.testing import assert_allclose

from salem import (
    GeoTiff,
    read_shapefile,
    read_shapefile_to_grid,
    sio,
    transform_geopandas,
    utils,
)
from salem.tests import (
    requires_cartopy,
    requires_dask,
    requires_geopandas,
    requires_matplotlib,
)
from salem.utils import get_demo_file

current_dir = Path(__file__).parent
testdir = current_dir / 'tmp'


def is_cartopy_rotated_working() -> bool:
    import pyproj
    from cartopy.crs import PlateCarree

    from salem.gis import proj_to_cartopy

    cp = pyproj.Proj(
        '+ellps=WGS84 +proj=ob_tran +o_proj=latlon '
        '+to_meter=0.0174532925199433 +o_lon_p=0.0 +o_lat_p=80.5 '
        '+lon_0=357.5 +no_defs'
    )
    cp = proj_to_cartopy(cp)

    out = PlateCarree().transform_points(cp, np.array([-20]), np.array([-9]))

    # Cartopy also had issues
    return np.allclose(
        out[0, 0], -22.243473889042903, atol=1e-5
    ) and np.allclose(out[0, 1], -0.06328365194179102, atol=1e-5)


@requires_geopandas
def create_dummy_shp(fname: Path | str) -> Path:
    import geopandas as gpd
    import shapely.geometry as shpg

    if isinstance(fname, str):
        fname = Path(fname)
    e_line = shpg.LinearRing([(1.5, 1), (2.0, 1.5), (1.5, 2.0), (1, 1.5)])
    i_line = shpg.LinearRing([(1.4, 1.4), (1.6, 1.4), (1.6, 1.6), (1.4, 1.6)])
    p1 = shpg.Polygon(e_line, [i_line])
    p2 = shpg.Polygon([(2.5, 1.3), (3.0, 1.8), (2.5, 2.3), (2, 1.8)])
    df = gpd.GeoDataFrame(crs='EPSG:4326', geometry=gpd.GeoSeries([p1, p2]))
    df['name'] = ['Polygon', 'Line']
    of = testdir / fname
    df.to_file(of)
    return of


def delete_test_dir() -> None:
    if testdir.exists():
        shutil.rmtree(testdir)


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        if not testdir.exists():
            testdir.mkdir(parents=True)

    def tearDown(self) -> None:
        delete_test_dir()

    def test_hash_cache_dir(self) -> None:
        h1 = utils._hash_cache_dir()
        h2 = utils._hash_cache_dir()
        assert h1 == h2

    def test_demofiles(self) -> None:
        assert utils.get_demo_file('dem_wgs84.nc').exists()
        with pytest.raises(FileNotFoundError):
            utils.get_demo_file('dummy')

    def test_read_colormap(self) -> None:
        cl = utils.read_colormap('topo') * 256
        assert_allclose(cl[4, :], (177, 242, 196))
        assert_allclose(cl[-1, :], (235, 233, 235))

        cl = utils.read_colormap('dem') * 256
        assert_allclose(cl[4, :], (153, 100, 43))
        assert_allclose(cl[-1, :], (255, 255, 255))

    def test_reduce(self) -> None:
        arr = [[1, 1, 2, 2], [1, 1, 2, 2]]
        assert_allclose(utils.reduce(arr, 1), arr)
        assert_allclose(utils.reduce(arr, 2), [[1, 2]])
        assert_allclose(utils.reduce(arr, 2, how=np.sum), [[4, 8]])

        arr = np.stack([arr, arr, arr])
        assert_allclose(arr.shape, (3, 2, 4))
        assert_allclose(utils.reduce(arr, 1), arr)
        assert_allclose(utils.reduce(arr, 2), [[[1, 2]], [[1, 2]], [[1, 2]]])
        assert_allclose(
            utils.reduce(arr, 2, how=np.sum), [[[4, 8]], [[4, 8]], [[4, 8]]]
        )
        arr[0, ...] = 0
        assert_allclose(
            utils.reduce(arr, 2, how=np.sum), [[[0, 0]], [[4, 8]], [[4, 8]]]
        )
        arr[1, ...] = 1
        assert_allclose(
            utils.reduce(arr, 2, how=np.sum), [[[0, 0]], [[4, 4]], [[4, 8]]]
        )


class TestIO(unittest.TestCase):
    def setUp(self) -> None:
        if not testdir.exists():
            testdir.mkdir(parents=True)

    def tearDown(self) -> None:
        delete_test_dir()

    @requires_geopandas
    def test_cache_working(self) -> None:
        f1 = 'f1.shp'
        f1 = create_dummy_shp(f1)
        cf1 = utils.cached_shapefile_path(f1)
        assert not cf1.exists()
        _ = read_shapefile(f1)
        assert not cf1.exists()
        _ = read_shapefile(f1, cached=True)
        assert cf1.exists()
        # nested calls
        assert cf1 == utils.cached_shapefile_path(cf1)

        # wait a bit
        time.sleep(0.1)
        f1 = create_dummy_shp(f1)
        cf2 = utils.cached_shapefile_path(f1)
        assert not cf1.exists()
        _ = read_shapefile(f1, cached=True)
        assert not cf1.exists()
        assert cf2.exists()
        df = read_shapefile(f1, cached=True)
        np.testing.assert_allclose(df.min_x, [1.0, 2.0])
        np.testing.assert_allclose(df.max_x, [2.0, 3.0])
        np.testing.assert_allclose(df.min_y, [1.0, 1.3])
        np.testing.assert_allclose(df.max_y, [2.0, 2.3])

        with pytest.raises(ValueError, match='File extension not recognised'):
            read_shapefile(Path('f1.sph'))
        with pytest.raises(ValueError, match='File extension not recognised'):
            utils.cached_shapefile_path(Path('f1.splash'))

    @requires_geopandas
    def test_read_to_grid(self) -> None:
        g = GeoTiff(utils.get_demo_file('hef_srtm.tif'))
        sf = utils.get_demo_file('Hintereisferner_UTM.shp')

        df1 = read_shapefile_to_grid(sf, g.grid)

        df2 = transform_geopandas(read_shapefile(sf), to_crs=g.grid)
        assert_allclose(
            df1.geometry[0].exterior.coords, df2.geometry[0].exterior.coords
        )

        # test for caching
        d = g.grid.to_dict()
        # change key ordering by chance
        d2 = dict((k, v) for k, v in d.items())  # noqa: C402

        from salem.sio import _memory_shapefile_to_grid, cached_shapefile_path

        shape_cpath = cached_shapefile_path(sf)
        res = _memory_shapefile_to_grid.call_and_shelve(
            shape_cpath, grid=g.grid, **d
        )
        try:
            h1 = res.timestamp
        except AttributeError:
            h1 = res.argument_hash
        res = _memory_shapefile_to_grid.call_and_shelve(
            shape_cpath, grid=g.grid, **d2
        )
        try:
            h2 = res.timestamp
        except AttributeError:
            h2 = res.argument_hash
        assert h1 == h2

    def test_notimevar(self) -> None:
        import xarray as xr

        da = xr.DataArray(np.arange(12).reshape(3, 4), dims=['lat', 'lon'])
        ds = da.to_dataset(name='var')

        t = sio.netcdf_time(ds)
        assert t is None


class TestSkyIsFalling(unittest.TestCase):
    @requires_matplotlib
    def test_projplot(self) -> None:
        # this caused many problems on fabien's laptop.
        # this is just to be sure that on your system, everything is fine

        import matplotlib.pyplot as plt
        import pyproj

        from salem.gis import check_crs, transform_proj

        pyproj.Proj(proj='latlong', datum='WGS84')
        plt.figure()
        plt.close()

        srs = '+units=m +proj=lcc +lat_1=29.0 +lat_2=29.0 +lat_0=29.0 +lon_0=89.8'

        proj_out = check_crs('EPSG:4326')
        proj_in = pyproj.Proj(srs, preserve_units=True)

        lon, _ = transform_proj(
            proj_in, proj_out, np.array(-2235000), np.array(-2235000)
        )
        np.testing.assert_allclose(lon, 70.75731, atol=1e-5)

    def test_gh_152(self) -> None:
        # https://github.com/fmaussion/salem/issues/152

        import xarray as xr

        da = xr.DataArray(
            np.arange(20).reshape(4, 5),
            dims=['lat', 'lon'],
            coords={
                'lat': np.linspace(0, 30, 4),
                'lon': np.linspace(-20, 20, 5),
            },
        )
        da.salem.roi()


class TestXarray(unittest.TestCase):
    def setUp(self) -> None:
        if not testdir.exists():
            testdir.mkdir(parents=True)

    def tearDown(self) -> None:
        delete_test_dir()

    @requires_dask
    def test_era(self) -> None:
        ds = sio.open_xr_dataset(get_demo_file('era_interim_tibet.nc')).chunk()
        assert ds.salem.x_dim == 'longitude'
        assert ds.salem.y_dim == 'latitude'

        dss = ds.salem.subset(ds=ds)
        assert dss.salem.grid == ds.salem.grid

        lon = 91.1
        lat = 31.1
        dss = ds.salem.subset(corners=((lon, lat), (lon, lat)), margin=1)

        assert len(dss.latitude) == 3
        assert len(dss.longitude) == 3

        np.testing.assert_almost_equal(dss.longitude, [90.0, 90.75, 91.5])

    def test_roi(self) -> None:
        import xarray as xr

        # Check that all attrs are preserved
        with sio.open_xr_dataset(get_demo_file('era_interim_tibet.nc')) as ds:
            ds.encoding = {'_FillValue': np.nan}
            ds['t2m'].encoding = {'_FillValue': np.nan}
            ds_ = ds.salem.roi(roi=np.ones_like(ds.t2m.to_numpy()[0, ...]))
            xr.testing.assert_identical(ds, ds_)
            assert ds.encoding == ds_.encoding
            assert ds.t2m.encoding == ds_.t2m.encoding

    @requires_geopandas  # because of the grid tests, more robust with GDAL
    def test_basic_wrf(self) -> None:
        import xarray as xr

        ds = sio.open_xr_dataset(get_demo_file('wrf_tip_d1.nc')).chunk()

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
            _ = t2.salem.grid

    @requires_dask
    def test_geo_em(self) -> None:
        for i in [1, 2, 3]:
            fg = get_demo_file('geo_em_d0{}_lambert.nc'.format(i))
            ds = sio.open_wrf_dataset(fg).chunk()
            assert 'Time' not in ds.dims
            assert 'time' in ds.dims
            assert 'south_north' in ds.dims
            assert 'south_north' in ds.coords

    @requires_geopandas  # because of the grid tests, more robust with GDAL
    def test_wrf(self) -> None:
        import xarray as xr

        ds = sio.open_wrf_dataset(get_demo_file('wrf_tip_d1.nc')).chunk()

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
            _ = t2.salem.grid

    @requires_dask
    def test_ncl_diagvars(self) -> None:
        import xarray as xr

        wf = get_demo_file('wrf_cropped.nc')
        ncl_out = get_demo_file('wrf_cropped_ncl.nc')

        w = sio.open_wrf_dataset(wf).chunk()
        nc = xr.open_dataset(ncl_out)

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=1e-6)

        ref = nc['SLP']
        tot = w['SLP']
        tot = tot.to_numpy()
        assert_allclose(ref, tot, rtol=1e-6)

        w = w.isel(time=1, south_north=slice(12, 16), west_east=slice(9, 16))
        nc = nc.isel(Time=1, south_north=slice(12, 16), west_east=slice(9, 16))

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=1e-6)

        ref = nc['SLP']
        tot = w['SLP']
        tot = tot.to_numpy()
        assert_allclose(ref, tot, rtol=1e-6)

        w = w.isel(bottom_top=slice(3, 5))
        nc = nc.isel(bottom_top=slice(3, 5))

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=1e-6)

        ref = nc['SLP']
        tot = w['SLP']
        tot = tot.to_numpy()
        assert_allclose(ref, tot, rtol=1e-6)

    @requires_dask
    def test_ncl_diagvars_compressed(self) -> None:
        rtol = 2e-5
        import xarray as xr

        wf = get_demo_file('wrf_cropped_compressed.nc')
        ncl_out = get_demo_file('wrf_cropped_ncl.nc')

        w = sio.open_wrf_dataset(wf).chunk()
        nc = xr.open_dataset(ncl_out)

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=rtol)

        ref = nc['SLP']
        tot = w['SLP'].data
        assert_allclose(ref, tot, rtol=rtol)

        w = w.isel(time=1, south_north=slice(12, 16), west_east=slice(9, 16))
        nc = nc.isel(Time=1, south_north=slice(12, 16), west_east=slice(9, 16))

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=rtol)

        ref = nc['SLP']
        tot = w['SLP']
        assert_allclose(ref, tot, rtol=rtol)

        w = w.isel(bottom_top=slice(3, 5))
        nc = nc.isel(bottom_top=slice(3, 5))

        ref = nc['TK']
        tot = w['TK']
        assert_allclose(ref, tot, rtol=rtol)

        ref = nc['SLP']
        tot = w['SLP']
        assert_allclose(ref, tot, rtol=rtol)

    @requires_dask
    def test_unstagger(self) -> None:
        wf = get_demo_file('wrf_cropped.nc')

        w = sio.open_wrf_dataset(wf).chunk()
        nc = sio.open_xr_dataset(wf).chunk()

        nc['PH_UNSTAGG'] = nc['P'] * 0.0
        uns = (
            nc['PH'].isel(bottom_top_stag=slice(0, -1)).to_numpy()
            + nc['PH']
            .isel(bottom_top_stag=slice(1, len(nc.bottom_top_stag)))
            .to_numpy()
        )
        nc['PH_UNSTAGG'].values = uns * 0.5

        assert_allclose(w['PH'], nc['PH_UNSTAGG'])

        # chunk
        v = w['PH'].chunk(
            {'time': 1, 'bottom_top': 6, 'south_north': 13, 'west_east': 13}
        )
        assert_allclose(v.mean(), nc['PH_UNSTAGG'].mean(), atol=1e-2)

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

        w['PH'].chunk()

    @requires_dask
    def test_unstagger_compressed(self) -> None:
        wf = get_demo_file('wrf_cropped.nc')
        wfc = get_demo_file('wrf_cropped_compressed.nc')

        w = sio.open_wrf_dataset(wf).chunk()
        wc = sio.open_wrf_dataset(wfc).chunk()

        assert_allclose(w['PH'], wc['PH'], rtol=0.003)

    @requires_dask
    def test_diagvars(self) -> None:
        wf = get_demo_file('wrf_d01_allvars_cropped.nc')
        w = sio.open_wrf_dataset(wf).chunk()

        # ws
        w['ws_ref'] = np.sqrt(w['U'] ** 2 + w['V'] ** 2)
        assert_allclose(w['ws_ref'], w['WS'])
        wcrop = w.isel(west_east=slice(4, 8), bottom_top=4)
        assert_allclose(wcrop['ws_ref'], wcrop['WS'])

    @requires_dask
    def test_diagvars_compressed(self) -> None:
        wf = get_demo_file('wrf_d01_allvars_cropped_compressed.nc')
        w = sio.open_wrf_dataset(wf).chunk()

        # ws
        w['ws_ref'] = np.sqrt(w['U'] ** 2 + w['V'] ** 2)
        assert_allclose(w['ws_ref'], w['WS'])
        wcrop = w.isel(west_east=slice(4, 8), bottom_top=4)
        assert_allclose(wcrop['ws_ref'], wcrop['WS'])

    @requires_dask
    def test_prcp(self) -> None:
        wf = get_demo_file('wrfout_d01.nc')

        w = sio.open_wrf_dataset(wf).chunk()
        nc = sio.open_xr_dataset(wf)

        nc['REF_PRCP_NC'] = nc['RAINNC'] * 0.0
        uns = (
            nc['RAINNC']
            .isel(Time=slice(1, len(nc.bottom_top_stag)))
            .to_numpy()
            - nc['RAINNC'].isel(Time=slice(0, -1)).to_numpy()
        )
        nc['REF_PRCP_NC'].to_numpy()[1:, ...] = (
            uns * 60 / 180.0
        )  # for three hours
        nc['REF_PRCP_NC'].to_numpy()[0, ...] = np.nan

        nc['REF_PRCP_C'] = nc['RAINC'] * 0.0
        uns = (
            nc['RAINC'].isel(Time=slice(1, len(nc.bottom_top_stag))).to_numpy()
            - nc['RAINC'].isel(Time=slice(0, -1)).to_numpy()
        )
        nc['REF_PRCP_C'].to_numpy()[1:, ...] = (
            uns * 60 / 180.0
        )  # for three hours
        nc['REF_PRCP_C'].to_numpy()[0, ...] = np.nan

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
            assert ~np.any(np.isfinite(wn['PRCP' + suf].values))

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
            assert ~np.any(np.isfinite(wn['PRCP' + suf].values))

    @requires_dask
    def test_prcp_compressed(self) -> None:
        wf = get_demo_file('wrfout_d01.nc')
        wfc = get_demo_file('wrfout_d01_compressed.nc')

        w = sio.open_wrf_dataset(wf).chunk().isel(time=slice(1, -1))
        wc = sio.open_wrf_dataset(wfc).chunk().isel(time=slice(1, -1))

        for suf in ['_NC', '_C', '']:
            assert_allclose(w['PRCP' + suf], wc['PRCP' + suf], atol=0.0003)

    @requires_geopandas  # because of the grid tests, more robust with GDAL
    def test_transform_logic(self) -> None:
        # This is just for the naming and dim logic, the rest is tested elsewh
        ds1 = sio.open_wrf_dataset(get_demo_file('wrfout_d01.nc')).chunk()
        ds2 = sio.open_wrf_dataset(get_demo_file('wrfout_d01.nc')).chunk()

        # 2darray case
        t2 = ds2.T2.isel(time=1)
        with pytest.raises(ValueError):
            ds1.salem.transform_and_add(t2.values, grid=t2.salem.grid)

        ds1.salem.transform_and_add(
            t2.values, grid=t2.salem.grid, name='t2_2darr'
        )
        assert 't2_2darr' in ds1
        assert_allclose(
            ds1.t2_2darr.coords['south_north'], t2.coords['south_north']
        )
        assert_allclose(
            ds1.t2_2darr.coords['west_east'], t2.coords['west_east']
        )
        assert ds1.salem.grid == ds1.t2_2darr.salem.grid

        # 3darray case
        t2 = ds2.T2
        ds1.salem.transform_and_add(
            t2.values, grid=t2.salem.grid, name='t2_3darr'
        )
        assert 't2_3darr' in ds1
        assert_allclose(
            ds1.t2_3darr.coords['south_north'], t2.coords['south_north']
        )
        assert_allclose(
            ds1.t2_3darr.coords['west_east'], t2.coords['west_east']
        )
        assert 'time' in ds1.t2_3darr.coords

        # dataarray case
        ds1.salem.transform_and_add(t2, name='NEWT2')
        assert 'NEWT2' in ds1
        assert_allclose(ds1.NEWT2, ds1.T2)
        assert_allclose(
            ds1.t2_3darr.coords['south_north'], t2.coords['south_north']
        )
        assert_allclose(
            ds1.t2_3darr.coords['west_east'], t2.coords['west_east']
        )
        assert 'time' in ds1.t2_3darr.coords

        # dataset case
        ds1.salem.transform_and_add(
            ds2[['RAINC', 'RAINNC']],
            name={'RAINC': 'PRCPC', 'RAINNC': 'PRCPNC'},
        )
        assert 'PRCPC' in ds1
        assert_allclose(ds1.PRCPC, ds1.RAINC)
        assert 'time' in ds1.PRCPNC.coords

        # what happens with external data?
        dse = sio.open_xr_dataset(get_demo_file('era_interim_tibet.nc'))
        out = ds1.salem.transform(dse.t2m, interp='linear')
        assert_allclose(out.coords['south_north'], t2.coords['south_north'])
        assert_allclose(out.coords['west_east'], t2.coords['west_east'])

    @requires_geopandas
    def test_lookup_transform(self) -> None:
        dsw = sio.open_wrf_dataset(get_demo_file('wrfout_d01.nc')).chunk()
        dse = sio.open_xr_dataset(
            get_demo_file('era_interim_tibet.nc')
        ).chunk()
        out = dse.salem.lookup_transform(dsw.T2C.isel(time=0), method=len)
        # qualitative tests (quantitative testing done elsewhere)
        assert out[0, 0] == 0
        assert out.mean() > 1

        dsw = sio.open_wrf_dataset(get_demo_file('wrfout_d01.nc'))
        dse = sio.open_xr_dataset(get_demo_file('era_interim_tibet.nc'))
        _, lut = dse.salem.lookup_transform(
            dsw.T2C.isel(time=0), method=len, return_lut=True
        )
        out2 = dse.salem.lookup_transform(
            dsw.T2C.isel(time=0), method=len, lut=lut
        )
        # qualitative tests (quantitative testing done elsewhere)
        assert_allclose(out, out2)

    @requires_dask
    def test_full_wrf_wfile(self) -> None:
        from salem.wrftools import var_classes

        # TODO: these tests are qualitative and should be compared against ncl
        f = get_demo_file('wrf_d01_allvars_cropped.nc')
        ds = sio.open_wrf_dataset(f).chunk()

        # making a repr was causing trouble because of the small chunks
        _ = ds.__repr__()

        # just check that the data is here
        var_classes = copy.deepcopy(var_classes)
        for vn in var_classes:
            _ = ds[vn].to_numpy()
            dss = ds.isel(
                west_east=slice(2, 6),
                south_north=slice(2, 5),
                bottom_top=slice(0, 15),
            )
            _ = dss[vn].to_numpy()
            dss = ds.isel(west_east=1, south_north=2, bottom_top=3, time=2)
            _ = dss[vn].to_numpy()

        # some chunking experiments
        v = ds.WS.chunk(
            {'time': 2, 'bottom_top': 1, 'south_north': 4, 'west_east': 5}
        )
        assert_allclose(v.mean(), ds.WS.mean(), atol=1e-3)
        ds = ds.isel(time=slice(1, 4))
        v = ds.PRCP.chunk({'time': 1, 'south_north': 2, 'west_east': 2})
        assert_allclose(v.mean(), ds.PRCP.mean())
        assert_allclose(v.max(), ds.PRCP.max())

    @requires_dask
    def test_full_wrf_wfile_compressed(self) -> None:
        from salem.wrftools import var_classes

        # TODO: these tests are qualitative and should be compared against ncl
        f = get_demo_file('wrf_d01_allvars_cropped_compressed.nc')
        ds = sio.open_wrf_dataset(f).chunk()

        # making a repr was causing trouble because of the small chunks
        _ = ds.__repr__()

        # just check that the data is here
        var_classes = copy.deepcopy(var_classes)
        for vn in var_classes:
            _ = ds[vn].to_numpy()
            dss = ds.isel(
                west_east=slice(2, 6),
                south_north=slice(2, 5),
                bottom_top=slice(0, 15),
            )
            _ = dss[vn].to_numpy()
            dss = ds.isel(west_east=1, south_north=2, bottom_top=3, time=2)
            _ = dss[vn].to_numpy()

        # some chunking experiments
        v = ds.WS.chunk(
            {'time': 2, 'bottom_top': 1, 'south_north': 4, 'west_east': 5}
        )
        assert_allclose(v.mean(), ds.WS.mean(), atol=1e-3)
        ds = ds.isel(time=slice(1, 4))
        v = ds.PRCP.chunk({'time': 1, 'south_north': 2, 'west_east': 2})
        assert_allclose(v.mean(), ds.PRCP.mean())
        assert_allclose(v.max(), ds.PRCP.max())

    @requires_dask
    def test_3d_interp(self) -> None:
        f = get_demo_file('wrf_d01_allvars_cropped.nc')
        ds = sio.open_wrf_dataset(f).chunk()

        out = ds.salem.wrf_zlevel('Z', levels=6000.0)
        ref_2d = out * 0.0 + 6000.0
        assert_allclose(out, ref_2d)

        # this used to raise an error
        _ = out.isel(time=1)

        out = ds.salem.wrf_zlevel('Z', levels=[6000.0, 7000.0])
        assert_allclose(out.sel(z=6000.0), ref_2d)
        assert_allclose(out.sel(z=7000.0), ref_2d * 0.0 + 7000.0)
        assert np.all(np.isfinite(out))

        out = ds.salem.wrf_zlevel('Z')
        assert_allclose(out.sel(z=7500.0), ref_2d * 0.0 + 7500.0)

        out = ds.salem.wrf_plevel('PRESSURE', levels=400.0)
        ref_2d = out * 0.0 + 400.0
        assert_allclose(out, ref_2d)

        out = ds.salem.wrf_plevel('PRESSURE', levels=[400.0, 300.0])
        assert_allclose(out.sel(p=400.0), ref_2d)
        assert_allclose(out.sel(p=300.0), ref_2d * 0.0 + 300.0)

        out = ds.salem.wrf_plevel('PRESSURE')
        assert_allclose(out.sel(p=300.0), ref_2d * 0.0 + 300.0)
        assert np.any(~np.isfinite(out))

        out = ds.salem.wrf_plevel('PRESSURE', fill_value='extrapolate')
        assert_allclose(out.sel(p=300.0), ref_2d * 0.0 + 300.0)
        assert np.all(np.isfinite(out))

        ds = sio.open_wrf_dataset(get_demo_file('wrfout_d01.nc'))
        ws_h = ds.isel(time=1).salem.wrf_zlevel(
            'WS', levels=8000.0, use_multiprocessing=False
        )
        assert np.all(np.isfinite(ws_h))
        ws_h2 = ds.isel(time=1).salem.wrf_zlevel('WS', levels=8000.0)
        assert_allclose(ws_h, ws_h2)

    @requires_dask
    def test_3d_interp_compressed(self) -> None:
        f = get_demo_file('wrf_d01_allvars_cropped_compressed.nc')
        ds = sio.open_wrf_dataset(f).chunk()

        out = ds.salem.wrf_zlevel('Z', levels=6000.0)
        ref_2d = out * 0.0 + 6000.0
        assert_allclose(out, ref_2d)

        # this used to raise an error
        _ = out.isel(time=1)

        out = ds.salem.wrf_zlevel('Z', levels=[6000.0, 7000.0])
        assert_allclose(out.sel(z=6000.0), ref_2d)
        assert_allclose(out.sel(z=7000.0), ref_2d * 0.0 + 7000.0)
        assert np.all(np.isfinite(out))

        out = ds.salem.wrf_zlevel('Z')
        assert_allclose(out.sel(z=7500.0), ref_2d * 0.0 + 7500.0)

        out = ds.salem.wrf_plevel('PRESSURE', levels=400.0)
        ref_2d = out * 0.0 + 400.0
        assert_allclose(out, ref_2d)

        out = ds.salem.wrf_plevel('PRESSURE', levels=[400.0, 300.0])
        assert_allclose(out.sel(p=400.0), ref_2d)
        assert_allclose(out.sel(p=300.0), ref_2d * 0.0 + 300.0)

        out = ds.salem.wrf_plevel('PRESSURE')
        assert_allclose(out.sel(p=300.0), ref_2d * 0.0 + 300.0)
        assert np.any(~np.isfinite(out))

        out = ds.salem.wrf_plevel('PRESSURE', fill_value='extrapolate')
        assert_allclose(out.sel(p=300.0), ref_2d * 0.0 + 300.0)
        assert np.all(np.isfinite(out))

        ds = sio.open_wrf_dataset(get_demo_file('wrfout_d01.nc'))
        ws_h = ds.isel(time=1).salem.wrf_zlevel(
            'WS', levels=8000.0, use_multiprocessing=False
        )
        assert np.all(np.isfinite(ws_h))
        ws_h2 = ds.isel(time=1).salem.wrf_zlevel('WS', levels=8000.0)
        assert_allclose(ws_h, ws_h2)

    @requires_dask
    def test_mf_datasets(self) -> None:
        import xarray as xr

        # prepare the data
        f = get_demo_file('wrf_d01_allvars_cropped.nc')
        ds = xr.open_dataset(f)
        for i in range(4):
            dss = ds.isel(Time=[i])
            dss.to_netcdf(testdir / f'wrf_slice_{i}.nc')
            dss.close()
        ds = sio.open_wrf_dataset(f)
        dsm = sio.open_mf_wrf_dataset(testdir / 'wrf_slice_*.nc')

        assert_allclose(ds['RAINNC'], dsm['RAINNC'])
        assert_allclose(ds['GEOPOTENTIAL'], dsm['GEOPOTENTIAL'])
        assert_allclose(ds['T2C'], dsm['T2C'])
        assert 'PRCP' not in dsm.variables

        prcp_nc_r = dsm.RAINNC.salem.deacc(as_rate=False)
        assert prcp_nc_r.units == 'mm step-1'
        assert prcp_nc_r.description == 'TOTAL GRID SCALE PRECIPITATION'

        prcp_nc = dsm.RAINNC.salem.deacc()
        assert prcp_nc.units == 'mm h-1'
        assert prcp_nc.description == 'TOTAL GRID SCALE PRECIPITATION'

        assert_allclose(prcp_nc_r / 3, prcp_nc)

        # note that this is needed because there are variables which just
        # can't be computed lazily (i.e. prcp)
        fo = testdir / 'wrf_merged.nc'
        if fo.exists():
            fo.unlink()
        dsm = dsm[['RAINNC', 'RAINC']].load()
        dsm.to_netcdf(fo)
        dsm.close()
        dsm = sio.open_wrf_dataset(fo)
        assert_allclose(ds['PRCP'], dsm['PRCP'], rtol=1e-6)
        assert_allclose(
            prcp_nc, dsm['PRCP_NC'].isel(time=slice(1, 4)), rtol=1e-6
        )

    @requires_cartopy
    def test_metum(self) -> None:
        if not sio.is_rotated_proj_working():
            with pytest.raises(RuntimeError):
                sio.open_metum_dataset(get_demo_file('rotated_grid.nc'))
            return

        ds = sio.open_metum_dataset(get_demo_file('rotated_grid.nc'))

        # One way
        mylons, mylats = ds.salem.grid.ll_coordinates
        assert_allclose(mylons, ds.longitude_t, atol=1e-7)
        assert_allclose(mylats, ds.latitude_t, atol=1e-7)

        # Round trip
        i, j = ds.salem.grid.transform(mylons, mylats)
        ii, jj = ds.salem.grid.ij_coordinates
        assert_allclose(i, ii, atol=1e-7)
        assert_allclose(j, jj, atol=1e-7)

        # Cartopy
        if not is_cartopy_rotated_working():
            return

        from cartopy.crs import PlateCarree

        from salem.gis import proj_to_cartopy

        cp = proj_to_cartopy(ds.salem.grid.proj)

        xx, yy = ds.salem.grid.xy_coordinates
        out = PlateCarree().transform_points(cp, xx.flatten(), yy.flatten())
        assert_allclose(out[:, 0].reshape(ii.shape), ds.longitude_t, atol=1e-7)
        assert_allclose(out[:, 1].reshape(ii.shape), ds.latitude_t, atol=1e-7)

        # Round trip
        out = cp.transform_points(
            PlateCarree(),
            ds.longitude_t.to_numpy().flatten(),
            ds.latitude_t.to_numpy().flatten(),
        )
        assert_allclose(out[:, 0].reshape(ii.shape), xx, atol=1e-7)
        assert_allclose(out[:, 1].reshape(ii.shape), yy, atol=1e-7)


class TestGeogridSim(unittest.TestCase):
    @requires_geopandas
    def test_lambert(self) -> None:
        from salem.wrftools import geogrid_simulator

        g, _ = geogrid_simulator(get_demo_file('namelist_lambert.wps'))

        assert len(g) == 3

        for i in [1, 2, 3]:
            fg = get_demo_file('geo_em_d0{}_lambert.nc'.format(i))
            with netCDF4.Dataset(fg) as nc:
                nc.set_auto_mask(False)
                lon, lat = g[i - 1].ll_coordinates
                assert_allclose(lon, nc['XLONG_M'][0, ...], atol=1e-4)
                assert_allclose(lat, nc['XLAT_M'][0, ...], atol=1e-4)

    @requires_geopandas
    def test_lambert_tuto(self) -> None:
        from salem.wrftools import geogrid_simulator

        g, _ = geogrid_simulator(get_demo_file('namelist_tutorial.wps'))

        assert len(g) == 1

        fg = get_demo_file('geo_em.d01_tutorial.nc')
        with netCDF4.Dataset(fg) as nc:
            nc.set_auto_mask(False)
            lon, lat = g[0].ll_coordinates
            assert_allclose(lon, nc['XLONG_M'][0, ...], atol=1e-4)
            assert_allclose(lat, nc['XLAT_M'][0, ...], atol=1e-4)

    @requires_geopandas
    def test_mercator(self) -> None:
        from salem.wrftools import geogrid_simulator

        g, _ = geogrid_simulator(get_demo_file('namelist_mercator.wps'))

        assert len(g) == 4

        for i in [1, 2, 3, 4]:
            fg = get_demo_file('geo_em_d0{}_mercator.nc'.format(i))
            with netCDF4.Dataset(fg) as nc:
                nc.set_auto_mask(False)
                lon, lat = g[i - 1].ll_coordinates
                assert_allclose(lon, nc['XLONG_M'][0, ...], atol=1e-4)
                assert_allclose(lat, nc['XLAT_M'][0, ...], atol=1e-4)

    @requires_geopandas
    def test_polar(self) -> None:
        from salem.wrftools import geogrid_simulator

        g, _ = geogrid_simulator(get_demo_file('namelist_polar.wps'))

        assert len(g) == 2

        for i in [1, 2]:
            fg = get_demo_file('geo_em_d0{}_polarstereo.nc'.format(i))
            with netCDF4.Dataset(fg) as nc:
                nc.set_auto_mask(False)
                lon, lat = g[i - 1].ll_coordinates
                assert_allclose(lon, nc['XLONG_M'][0, ...], atol=5e-3)
                assert_allclose(lat, nc['XLAT_M'][0, ...], atol=5e-3)
