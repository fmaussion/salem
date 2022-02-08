"""
Some useful functions
"""
from __future__ import division

import io
import os
import shutil
import zipfile
from collections import OrderedDict

import numpy as np
from joblib import Memory
from salem import (cache_dir, sample_data_dir, sample_data_gh_commit,
                   download_dir, python_version)
from six.moves.urllib.request import urlretrieve, urlopen


def _hash_cache_dir():
    """Get the path to the right cache directory.

    We need to make sure that cached files correspond to the same
    environment. To this end we make a unique directory hash, depending on the
    version and location of several packages we thought are important
    (because they change often, or because conda versions give different
    results than pip versions).

    Returns
    -------
    path to the dir
    """
    import hashlib

    out = OrderedDict(python_version=python_version)

    try:
        import shapely
        out['shapely_version'] = shapely.__version__
        out['shapely_file'] = shapely.__file__
    except ImportError:
        pass
    try:
        import fiona
        out['fiona_version'] = fiona.__version__
        out['fiona_file'] = fiona.__file__
    except ImportError:
        pass
    try:
        import pandas
        out['pandas_version'] = pandas.__version__
        out['pandas_file'] = pandas.__file__
    except ImportError:
        pass
    try:
        import geopandas
        out['geopandas_version'] = geopandas.__version__
        out['geopandas_file'] = geopandas.__file__
    except ImportError:
        pass
    try:
        import osgeo
        out['osgeo_version'] = osgeo.__version__
        out['osgeo_file'] = osgeo.__file__
    except ImportError:
        pass
    try:
        import pyproj
        out['pyproj_version'] = pyproj.__version__
        out['pyproj_file'] = pyproj.__file__
    except ImportError:
        pass
    try:
        import salem
        out['salem_version'] = salem.__version__
        out['salem_file'] = salem.__file__
    except ImportError:
        pass

    # ok, now make a dummy str that we will hash
    strout = ''
    for k, v in out.items():
        strout += k + v
    strout = 'salem_hash_' + hashlib.md5(strout.encode()).hexdigest()
    dirout = os.path.join(cache_dir, 'cache', strout)
    return dirout


hash_cache_dir = _hash_cache_dir()
try:
    memory = Memory(location=hash_cache_dir + '_joblib', verbose=0)
except TypeError:
    # https://github.com/fmaussion/salem/issues/130
    memory = Memory(cachedir=hash_cache_dir + '_joblib', verbose=0)

# A series of variables and dimension names that Salem will understand
valid_names = dict()
valid_names['x_dim'] = ['west_east', 'lon', 'longitude', 'longitudes', 'lons',
                        'xlong', 'xlong_m', 'dimlon', 'x', 'lon_3', 'long',
                        'phony_dim_0', 'eastings', 'easting', 'nlon', 'nlong',
                        'grid_longitude_t']
valid_names['y_dim'] = ['south_north', 'lat', 'latitude', 'latitudes', 'lats',
                        'xlat', 'xlat_m', 'dimlat', 'y','lat_3', 'phony_dim_1',
                        'northings', 'northing', 'nlat', 'grid_latitude_t']
valid_names['z_dim'] = ['levelist','level', 'pressure', 'press', 'zlevel', 'z',
                        'bottom_top']
valid_names['t_dim'] = ['time', 'times', 'xtime']

valid_names['lon_var'] = ['lon', 'longitude', 'longitudes', 'lons', 'long']
valid_names['lat_var'] = ['lat', 'latitude', 'latitudes', 'lats']
valid_names['time_var'] = ['time', 'times']

sample_data_gh_repo = 'fmaussion/salem-sample-data'
nearth_base = 'http://shadedrelief.com/natural3/ne3_data/'


def str_in_list(l1, l2):
    """Check if one element of l1 is in l2 and if yes, returns the name of
    that element in a list (could be more than one.

    Examples
    --------
    >>> print(str_in_list(['time', 'lon'], ['temp','time','prcp']))
    ['time']
    >>> print(str_in_list(['time', 'lon'], ['temp','time','prcp','lon']))
    ['time', 'lon']
    """
    return [i for i in l1 if i.lower() in l2]


def empty_cache():
    """Empty salem's cache directory."""

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir)


def cached_shapefile_path(fpath):
    """Checks if a shapefile is cached and returns the corresponding path.

    This function checks for the last time the file has changed,
    so it should be safe to use.
    """

    p, ext = os.path.splitext(fpath)

    if ext.lower() == '.p':
        # No need to recache pickled files (this is for nested calls)
        return fpath

    if ext.lower() != '.shp':
        raise ValueError('File extension not recognised: {}'.format(ext))

    # Cached directory and file
    cp = os.path.commonprefix([cache_dir, p])
    cp = os.path.join(cache_dir, hash_cache_dir + '_shp',
                      os.path.relpath(p, cp))
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


def _urlretrieve(url, ofile, *args, **kwargs):
    """Wrapper for urlretrieve which overwrites."""

    try:
        return urlretrieve(url, ofile, *args, **kwargs)
    except:
        if os.path.exists(ofile):
            os.remove(ofile)
        # try to make the thing more robust with a second shot
        try:
            return urlretrieve(url, ofile, *args, **kwargs)
        except:
            if os.path.exists(ofile):
                os.remove(ofile)
            raise


def download_demo_files():
    """Checks if the demo data is already on the cache and downloads it.

    Borrowed from OGGM.
    """

    master_zip_url = 'https://github.com/%s/archive/%s.zip' % \
                     (sample_data_gh_repo, sample_data_gh_commit)
    ofile = os.path.join(cache_dir,
                         'salem-sample-data-%s.zip' % sample_data_gh_commit)
    odir = os.path.join(cache_dir)

    # download only if necessary
    if not os.path.exists(ofile):
        print('Downloading salem-sample-data...')
        _urlretrieve(master_zip_url, ofile)

        # Trying to make the download more robust
        try:
            with zipfile.ZipFile(ofile) as zf:
                zf.extractall(odir)
        except zipfile.BadZipfile:
            # try another time
            if os.path.exists(ofile):
                os.remove(ofile)
            _urlretrieve(master_zip_url, ofile)
            with zipfile.ZipFile(ofile) as zf:
                zf.extractall(odir)

    # list of files for output
    out = dict()
    for root, directories, filenames in os.walk(sample_data_dir):
        for filename in filenames:
            out[filename] = os.path.join(root, filename)

    return out


def get_demo_file(fname):
    """Returns the path to the desired demo file."""

    d = download_demo_files()
    if fname in d:
        return d[fname]
    else:
        return None


def get_natural_earth_file(res='lr'):
    """Returns the path to the desired natural earth file.

    http://www.shadedrelief.com/natural3/pages/textures.html

    Parameters
    ----------
    res : str
       'lr' or 'hr' (low res or high res)
    """

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    if res == 'lr':
        return get_demo_file('natural_earth_lr.jpg')
    elif res == 'mr':
        urlpath = nearth_base + '8192/textures/2_no_clouds_8k.jpg'
    elif res == 'hr':
        urlpath = nearth_base + '16200/textures/2_no_clouds_16k.jpg'
    ofile = os.path.join(download_dir, 'natural_earth_' + res+ '.jpg')

    # download only if necessary
    if not os.path.exists(ofile):
        print('Downloading Natural Earth ' + res + '...')
        _urlretrieve(urlpath, ofile)

    return ofile


@memory.cache
def read_colormap(name):
    """Reads a colormap from the custom files in Salem."""

    path = get_demo_file(name + '.c3g')

    out = []
    with open(path, 'r') as file:
        for line in file:
            if 'rgb(' not in line:
                continue
            line = line.split('(')[-1].split(')')[0]
            out.append([float(n) for n in line.split(',')])

    return np.asarray(out).astype(float) / 256.


@memory.cache
def joblib_read_img_url(url):
    """Prevent to re-download from GoogleStaticMap if it was done before"""

    from matplotlib.image import imread
    fd = urlopen(url, timeout=10)
    return imread(io.BytesIO(fd.read()))


def nice_scale(mapextent, maxlen=0.15):
    """Returns a nice number for a legend scale of a map.

    Parameters
    ----------
    mapextent : float
        the total extent of the map
    maxlen : float
        from 0 to 1, the maximum relative length allowed for the scale

    Examples
    --------
    >>> print(nice_scale(140))
    20.0
    >>> print(nice_scale(140, maxlen=0.5))
    50.0
    """
    d = np.array([1, 2, 5])
    e = (np.ones(12) * 10) ** (np.arange(12)-5)
    candidates = np.matmul(e[:, None],  d[None, :]).flatten()
    return np.max(candidates[candidates / mapextent <= maxlen])


def reduce(arr, factor=1, how=np.mean):
    """Reduces an array's size by a given factor.

    The reduction can be done by any reduction function (default is mean).

    Parameters
    ----------
    arr : ndarray
        an array of at least 2 dimensions (the reduction is done on the two
        last dimensions).
    factor : int
        the factor to apply for reduction (must be a divider of the original
        axis dimension!).
    how : func
        the reduction function

    Returns
    -------
    the reduced array
    """
    arr = np.asarray(arr)
    shape = list(arr.shape)
    newshape = shape[:-2] + [np.round(shape[-2] / factor).astype(int), factor,
                             np.round(shape[-1] / factor).astype(int), factor]
    return how(how(arr.reshape(*newshape), axis=len(newshape)-3),
               axis=len(newshape)-2)
