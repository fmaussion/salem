"""Some useful functions

Copyright: Fabien Maussion, 2014-2016

License: GPLv3+
"""
from __future__ import division

import io
import json
import os
import shutil
import time
import zipfile

import numpy as np
from joblib import Memory
from salem import cache_dir, download_dir, python_version
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlretrieve, urlopen

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

sample_data_gh_repo = 'fmaussion/salem-sample-data'
nearth_base = 'http://naturalearth.springercarto.com/ne3_data/'


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

    master_sha_url = 'https://api.github.com/repos/%s/commits/master' % \
                     sample_data_gh_repo
    master_zip_url = 'https://github.com/%s/archive/master.zip' % \
                     sample_data_gh_repo
    ofile = os.path.join(cache_dir, 'salem-sample-data.zip')
    shafile = os.path.join(cache_dir, 'salem-sample-data-commit.txt')
    odir = os.path.join(cache_dir)

    # a file containing the online's file's hash and the time of last check
    if os.path.exists(shafile):
        with open(shafile, 'r') as sfile:
            local_sha = sfile.read().strip()
        last_mod = os.path.getmtime(shafile)
    else:
        # very first download
        local_sha = '0000'
        last_mod = 0

    # test only every hour
    if time.time() - last_mod > 3600:
        write_sha = True
        try:
            # this might fail with HTTP 403 when server overload
            resp = urlopen(master_sha_url)

            # following try/finally is just for py2/3 compatibility
            # https://mail.python.org/pipermail/python-list/2016-March/704073.html
            try:
                json_str = resp.read().decode('utf-8')
            finally:
                resp.close()
            json_obj = json.loads(json_str)
            master_sha = json_obj['sha']
            # if not same, delete entire dir
            if local_sha != master_sha:
                empty_cache()
        except (HTTPError, URLError):
            master_sha = 'error'
    else:
        write_sha = False

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

    # sha did change, replace
    if write_sha:
        with open(shafile, 'w') as sfile:
            sfile.write(master_sha)

    # list of files for output
    out = dict()
    sdir = os.path.join(cache_dir, 'salem-sample-data-master')
    for root, directories, filenames in os.walk(sdir):
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

    return np.asarray(out).astype(np.float) / 256.


@memory.cache
def joblib_read_img_url(url):
    """Prevent to re-download from GoogleStaticMap if it was done before"""

    from matplotlib.image import imread
    fd = urlopen(url, timeout=10)
    return imread(io.BytesIO(fd.read()))
