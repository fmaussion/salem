from __future__ import division

import copy
import os
import shutil

import numpy as np
import pytest

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import shapely.geometry as shpg
    import geopandas as gpd
except ImportError:
    pass

from salem.graphics import ExtendedNorm, DataLevels, Map, get_cmap, shapefiles
from salem import Grid, wgs84, mercator_grid, GeoNetcdf, \
    read_shapefile_to_grid, GeoTiff, GoogleCenterMap, GoogleVisibleMap
from salem.utils import get_demo_file
from salem.tests import requires_matplotlib

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(current_dir, 'tmp')


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
    df.crs = {'init': 'epsg:4326'}
    df.to_file(of)
    return of


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images',
                               tolerance=5)
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_datalevels_visu_h():
    a = np.array([-1., 0., 1.1, 1.9, 9.])
    cm = mpl.cm.get_cmap('RdYlBu_r')

    dl = DataLevels(a, cmap=cm, levels=[0, 1, 2, 3])

    fig, ax = plt.subplots(1)
    dl.visualize(ax=ax, orientation='horizontal', add_values=True)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_datalevels_visu_v():
    a = np.array([-1., 0., 1.1, 1.9, 9.])
    cm = mpl.cm.get_cmap('RdYlBu_r')

    dl = DataLevels(a.reshape((5, 1)), cmap=cm, levels=[0, 1, 2, 3])

    fig, ax = plt.subplots(1)
    dl.visualize(ax=ax, orientation='vertical', add_values=True)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
    g1 = Grid(nxny=(5, 4), dxdy=(1, -1), ul_corner=(-1, 3), proj=wgs84,
              pixel_ref='corner')
    c1 = Map(g1, ny=4, countries=False)

    # LL Corner
    g2 = Grid(nxny=(5, 4), dxdy=(1, 1), ll_corner=(-1, -1), proj=wgs84,
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_contourf():
    a = np.zeros((4, 5))
    a[0, 0] = -1
    a[1, 1] = 1.1
    a[2, 2] = 2.2
    a[2, 4] = 1.9
    a[3, 3] = 9

    # UL Corner
    g = Grid(nxny=(5, 4), dxdy=(1, -1), ul_corner=(-1, 3), proj=wgs84,
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

    fig, ax = plt.subplots(1)
    c.visualize(ax=ax)
    fig.tight_layout()

    # remove it
    c.set_contourf()
    c.set_contour()

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_merca_map():
    grid = mercator_grid(center_ll=(11.38, 47.26),
                         extent=(2000000, 2000000))

    m1 = Map(grid)

    grid = mercator_grid(center_ll=(11.38, 47.26),
                         extent=(2000000, 2000000),
                         order='ul')
    m2 = Map(grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    m1.visualize(ax=ax1, addcbar=False)
    m2.visualize(ax=ax2, addcbar=False)
    plt.tight_layout()

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_oceans():
    f = os.path.join(get_demo_file('wrf_tip_d1.nc'))
    grid = GeoNetcdf(f).grid
    m = Map(grid, countries=False)
    m.set_shapefile(rivers=True, linewidths=2)
    m.set_shapefile(oceans=True, edgecolor='k', linewidth=3)

    fig, ax = plt.subplots(1, 1)
    m.visualize(ax=ax, addcbar=False)
    plt.tight_layout()

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_geometries():
    # UL Corner
    g = Grid(nxny=(5, 4), dxdy=(10, 10), ll_corner=(-20, -15), proj=wgs84,
             pixel_ref='corner')
    c = Map(g, ny=4)
    c.set_lonlat_contours(interval=10., colors='crimson')

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

    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, addcbar=False)
    plt.tight_layout()

    c.set_geometry()
    assert len(c._geometries) == 0

    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_text():
    # UL Corner
    g = Grid(nxny=(5, 4), dxdy=(10, 10), ll_corner=(-20, -15), proj=wgs84,
             pixel_ref='corner')
    c = Map(g, ny=4, countries=False)
    c.set_lonlat_contours(interval=5., colors='crimson')

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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_hef_default_spline():
    grid = mercator_grid(center_ll=(10.76, 46.798444),
                         extent=(10000, 7000))
    c = Map(grid, countries=False)
    c.set_lonlat_contours(interval=10)
    c.set_shapefile(get_demo_file('Hintereisferner_UTM.shp'))
    c.set_topography(get_demo_file('hef_srtm.tif'))

    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, addcbar=False, title='Default: spline deg 3')
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_hef_from_array():
    grid = mercator_grid(center_ll=(10.76, 46.798444),
                         extent=(10000, 7000))
    c = Map(grid, countries=False)
    c.set_lonlat_contours(interval=10)
    c.set_shapefile(get_demo_file('Hintereisferner_UTM.shp'))

    dem = GeoTiff(get_demo_file('hef_srtm.tif'))
    mytopo = dem.get_vardata()
    c.set_topography(mytopo, crs=dem.grid, interp='spline')

    fig, ax = plt.subplots(1, 1)
    c.visualize(ax=ax, addcbar=False, title='From array')
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images',
                               tolerance=5)
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_gmap():
    g = GoogleCenterMap(center_ll=(10.762660, 46.794221), zoom=13,
                        size_x=640, size_y=640)

    m = Map(g.grid, countries=False, nx=640)
    m.set_lonlat_contours(interval=0.025)
    m.set_shapefile(get_demo_file('Hintereisferner.shp'),
                    linewidths=2, edgecolor='darkred')
    m.set_rgb(g.get_vardata())

    fig, ax = plt.subplots(1, 1)
    m.visualize(ax=ax, addcbar=False)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
def test_gmap_transformed():
    dem = GeoTiff(get_demo_file('hef_srtm.tif'))
    dem.set_subset(margin=-100)

    dem = mercator_grid(center_ll=(10.76, 46.798444),
                        extent=(10000, 7000))

    i, j = dem.ij_coordinates
    g = GoogleVisibleMap(x=i, y=j, src=dem, size_x=500, size_y=400)
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
    smap.set_shapefile(shape=shdf, color='grey', linewidth=3)
    smap.set_points(91.1, 29.6)
    smap.set_text(91.2, 29.7, 'Lhasa', fontsize=17)
    smap.set_data(ds.T2.isel(Time=1)-273.15, crs=ds.salem.grid)

    fig, ax = plt.subplots(1, 1)
    smap.visualize(ax=ax)
    plt.tight_layout()
    return fig


@requires_matplotlib
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
@pytest.mark.mpl_image_compare(baseline_dir='baseline_images')
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
