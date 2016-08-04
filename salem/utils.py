"""Some useful functions

Copyright: Fabien Maussion, 2014-2015

License: GPLv3+
"""
from __future__ import division
from six.moves.urllib.request import urlretrieve, urlopen

# Builtins
import os
import io
import shutil
import pickle
import zipfile

# External libs
import numpy as np
from joblib import Memory
try:
    import geopandas as gpd
except ImportError:
    pass
from matplotlib.image import imread

# Locals
from salem import cache_dir
from salem import python_version
from salem import transform_geopandas

# Joblib
memory = Memory(cachedir=cache_dir, verbose=0)

# A series of variables and dimension names that Salem will understand
valid_names = dict()
valid_names['x_dim'] = ['west_east', 'lon', 'longitude', 'longitudes', 'lons',
                        'xlong', 'xlong_m', 'dimlon', 'x', 'lon_3', 'long',
                        'phony_dim_0']
valid_names['y_dim'] = ['south_north', 'lat', 'latitude', 'latitudes', 'lats',
                        'xlat', 'xlat_m', 'dimlat', 'y','lat_3', 'phony_dim_1']
valid_names['z_dim'] = ['levelist','level', 'pressure', 'press', 'zlevel', 'z']
valid_names['t_dim'] = ['time', 'times', 'xtime']

valid_names['lon_var'] = ['lon', 'longitude', 'longitudes', 'lons', 'long']
valid_names['lat_var'] = ['lat', 'latitude', 'latitudes', 'lats']
valid_names['time_var'] = ['time', 'times']

gh_zip = 'https://github.com/fmaussion/salem-sample-data/archive/master.zip'


def str_in_list(l1, l2):
    """Check if one element of l1 is in l2 and if yes, returns the name of
    that element.

    Examples
    --------
    >>> print(str_in_list(['lat', 'lon'], ['time', 'times']))
    None
    >>> str_in_list(['Time', 'lat', 'lon'], ['time', 'times'])
    'Time'
    """
    vt = [i for i in l1 if i.lower() in l2]
    if len(vt) > 0:
        return vt[0]
    else:
        return None


def empty_cache():  # pragma: no cover
    """Empty salem's cache directory."""

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)


def cached_path(fpath):
    """Checks if a file is cached and returns the corresponding path.

    This function checks for the last time the file has changed,
    so it should be safe to use.
    """

    p, ext = os.path.splitext(fpath)

    if ext.lower() == '.p':
        # No need to recache pickled files (this is for possible nested calls)
        return fpath

    if ext.lower() != '.shp':
        raise ValueError('File extension not recognised: {}'.format(ext))

    # Cached directory and file
    cp = os.path.commonprefix([cache_dir, p])
    cp = os.path.join(cache_dir, python_version, os.path.relpath(p, cp))
    ct = '{:d}'.format(int(round(os.path.getmtime(fpath)*1000.)))
    of = os.path.join(cp, ct + '.p')
    if os.path.exists(cp):
        # We have to check if the file changed
        if os.path.exists(of):
            return of
        else:
            # the file has changed
            shutil.rmtree(cp)

    os.makedirs(cp)
    return of


def _download_demo_files():
    """Checks if the demo data is already on the cache and downloads it.

    Currently there's no check to see of the server file has changed: this
    is bad. In the mean time, empty_cache() will ensure that the files are
    up-to-date.
    """

    ofile = os.path.join(cache_dir, 'salem-sample-data.zip')
    odir = os.path.join(cache_dir)
    if not os.path.exists(ofile):  # pragma: no cover
        urlretrieve(gh_zip, ofile)
        with zipfile.ZipFile(ofile) as zf:
            zf.extractall(odir)

    out = dict()
    sdir = os.path.join(cache_dir, 'salem-sample-data-master', 'salem-test')
    for root, directories, filenames in os.walk(sdir):
        for filename in filenames:
            out[filename] = os.path.join(root, filename)
    return out


def get_demo_file(fname):
    """Returns the path to the desired demo file."""

    d = _download_demo_files()
    if fname in d:
        return d[fname]
    else:
        return None


def read_shapefile(fpath, cached=False):
    """Reads a shapefile using geopandas.

    Because reading a shapefile can take a long time, Salem provides a
    caching utility (cached=True). This will save a pickle of the shapefile
    in the cache directory ('.salem_cache' in the user directory).
    """

    _, ext = os.path.splitext(fpath)
    # TODO: remove this crs stuff when geopandas is uptated (> 0.1.1)
    # https://github.com/geopandas/geopandas/issues/199
    if ext.lower() in ['.shp', '.p']:
        if cached:
            cpath = cached_path(fpath)
            if os.path.exists(cpath):
                with open(cpath, 'rb') as f:
                    pick = pickle.load(f)
                out = pick['gpd']
                out.crs = pick['crs']
            else:
                out = read_shapefile(fpath, cached=False)
                pick = dict(gpd=out, crs=out.crs)
                with open(cpath, 'wb') as f:
                    pickle.dump(pick, f)
        else:
            out = gpd.read_file(fpath)
            out['min_x'] = [g.bounds[0] for g in out.geometry]
            out['max_x'] = [g.bounds[2] for g in out.geometry]
            out['min_y'] = [g.bounds[1] for g in out.geometry]
            out['max_y'] = [g.bounds[3] for g in out.geometry]
    else:
        raise ValueError('File extension not recognised: {}'.format(ext))

    return out


@memory.cache(ignore=['grid'])
def _memory_transform(shape_cpath, grid=None, grid_str=None):
    """Quick solution using joblib in order to not transform many times the
    same shape (usefull for Cleo).

    Since grid is a complex object joblib seemed to have trouble with it,
    so joblib is checking its cache according to grid_str while the job is
    done with grid.
    """

    shape = read_shapefile(shape_cpath, cached=True)
    e = grid.extent_in_crs(crs=shape.crs)
    p = np.nonzero(~((shape['min_x'] > e[1]) |
                     (shape['max_x'] < e[0]) |
                     (shape['min_y'] > e[3]) |
                     (shape['max_y'] < e[2])))
    shape = shape.iloc[p]
    shape = transform_geopandas(shape, to_crs=grid, inplace=True)
    return shape, shape.crs


def read_shapefile_to_grid(fpath, grid):
    """Same as read_shapefile but the shapefile is directly transformed to
    the desired grid. The whole thing is cached so that the second call will
    will be much faster.

    Parameters
    ----------
    fpath: path to the file
    grid: the arrival grid
    """

    # ensure it is a cached pickle (copy code smell)
    shape_cpath = cached_path(fpath)
    if not os.path.exists(shape_cpath):
        out = read_shapefile(fpath, cached=False)
        pick = dict(gpd=out, crs=out.crs)
        with open(shape_cpath, 'wb') as f:
            pickle.dump(pick, f)

    #TODO: remove this when new geopandas is out
    out, crs = _memory_transform(shape_cpath, grid=grid, grid_str=str(grid))
    out.crs = crs
    return out


@memory.cache
def joblib_read_url(url):
    """Prevent to re-download from GoogleStaticMap if it was done before"""

    fd = urlopen(url)
    return imread(io.BytesIO(fd.read()))