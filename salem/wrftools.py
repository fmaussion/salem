"""Diagnostic variables for WRF output.

Diagnostic variables are simply a subclass of FakeVariable that implement
__getitem__. See examples below.
"""
from __future__ import division
import copy

import numpy as np
import pyproj
from scipy.interpolate import interp1d
from netCDF4 import num2date
from pandas import to_datetime
from xarray.core import indexing

from salem import lazy_property, wgs84, gis

POOL = None


def _init_pool():
    global POOL
    if POOL is None:
        import multiprocessing as mp
        POOL = mp.Pool()


def dummy_func(*args):
    pass


class ScaledVar():

    def __init__(self, ncvar):
        self.ncvar = ncvar
        try:
            self.scale = ncvar.scale
        except AttributeError:
            self.scale = False
            
    def __enter__(self):
        self.ncvar.set_auto_scale(True)
        return self.ncvar

    def __exit__(self, type, value, traceback):
        self.ncvar.set_auto_scale(self.scale)


class Unstaggerer(object):
    """Duck NetCDF4.Variable class which "unstaggers" WRF variables.

     It looks for the staggered dimension and automatically unstaggers it.
     """

    def __init__(self, ncvar):
        """Instanciate.

        Parameters
        ----------
        ncvar: the netCDF variable to unstagger.
        """

        self.ncvar = ncvar

        # Attributes
        self.description = ncvar.description
        self.units = ncvar.units
        # Replace the dimension name
        dims = list(ncvar.dimensions)
        self.ds = np.nonzero(['_stag' in d for d in dims])[0][0]
        dims[self.ds] = dims[self.ds].replace('_stag', '')
        self.dimensions = dims
        shape = list(ncvar.shape)
        shape[self.ds] -=1
        self.shape = tuple(shape)

        # this is quickndirty, and probably wrong
        self.set_auto_maskandscale = dummy_func
        self.set_auto_scale = dummy_func
        attrs = list(ncvar.ncattrs())
        if 'add_offset' in attrs: attrs.remove('add_offset')
        if 'scale_factor' in attrs: attrs.remove('scale_factor')
        def filter_attrs():
            return attrs
        self.ncattrs = filter_attrs
        self.filters = ncvar.filters

        def _chunking():
            return self.shape
        self.chunking = _chunking

        for attr in self.ncattrs():
            setattr(self, attr, getattr(ncvar, attr))
        self.dtype = ncvar.dtype

        # Needed later
        self._ds_shape = ncvar.shape[self.ds]

    def getncattr(self, name):
        # dummy getncattrs
        return getattr(self, name)

    @staticmethod
    def can_do(ncvar):
        """Checks if the variable can be unstaggered.

        Parameters
        ----------
        ncvar: the netCDF variable candidate forunstagger.
        """
        return np.any(['_stag' in d for d in ncvar.dimensions])

    def __getitem__(self, item):
        """Override __getitem__."""

        # take care of ellipsis and other strange indexes
        item = list(indexing.expanded_indexer(item, len(self.dimensions)))

        # Slice to change
        was_scalar = False
        sl = item[self.ds]
        if np.isscalar(sl) and not isinstance(sl, slice):
            sl = slice(sl, sl+1)
            was_scalar = True

        # Ok, get the indexes right
        start = sl.start or 0
        stop = sl.stop or self._ds_shape
        if stop < 0:
            stop += self._ds_shape-1
        stop = np.clip(stop+1, 0, self._ds_shape)
        itemr = copy.deepcopy(item)
        if was_scalar:
            item[self.ds] = start
            itemr[self.ds] = start+1
        else:
            item[self.ds] = slice(start, stop-1)
            itemr[self.ds] = slice(start+1, stop)
        with ScaledVar(self.ncvar) as var:
            return 0.5*(var[tuple(item)] + var[tuple(itemr)])


class FakeVariable(object):
    """Duck NetCDF4.Variable class
    """
    def __init__(self, nc):
        self.name = self.__class__.__name__
        self.nc = nc

    @staticmethod
    def can_do():
        raise NotImplementedError()

    def _copy_attrs_from(self, ncvar):
        # copies the necessary nc attributes from a template variable
        attrs = list(ncvar.ncattrs())
        if 'add_offset' in attrs: attrs.remove('add_offset')
        if 'scale_factor' in attrs: attrs.remove('scale_factor')
        def filter_attrs():
            return attrs
        self.ncattrs = filter_attrs
        self.filters = ncvar.filters
        self.chunking = ncvar.chunking
        for attr in self.ncattrs():
            setattr(self, attr, getattr(ncvar, attr))
        self.dimensions = ncvar.dimensions
        self.dtype = ncvar.dtype
        self.set_auto_maskandscale = dummy_func
        self.set_auto_scale = dummy_func
        self.shape = ncvar.shape

    def getncattr(self, name):
        # dummy getncattrs
        return getattr(self, name)

    def __getitem__(self, item):
        raise NotImplementedError()


class T2C(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['T2'])
        self.units = 'C'
        self.description = '2m Temperature'

    @staticmethod
    def can_do(nc):
        return 'T2' in nc.variables

    def __getitem__(self, item):
        with ScaledVar(self.nc.variables['T2']) as var:
            return var[item] - 273.15


class AccumulatedVariable(FakeVariable):
    """Common logic for all accumulated variables."""

    def __init__(self, nc, accvn):
        FakeVariable.__init__(self, nc)
        self.accvn = accvn
        self._copy_attrs_from(nc.variables[self.accvn])
        # Needed later
        self._nel = nc.variables[self.accvn].shape[0]

    @lazy_property
    def _factor(self):
        # easy would be to have time step as variable
        vars = self.nc.variables
        if 'XTIME' in vars:
            dt_minutes = vars['XTIME'][1] - vars['XTIME'][0]
        elif 'xtime' in vars:
            dt_minutes = vars['xtime'][1] - vars['xtime'][0]
        elif 'time' in vars:
            var = vars['time']
            nctime = num2date(var[0:2], var.units)
            dt_minutes = np.asarray(nctime[1] - nctime[0])
            dt_minutes = dt_minutes.astype('timedelta64[m]').astype(float)
        else:
            # ok, parse time
            time = []
            stimes = vars['Times'][0:2]
            for t in stimes:
                time.append(to_datetime(t.tobytes().decode(),
                                        errors='raise',
                                        format='%Y-%m-%d_%H:%M:%S'))
            dt_minutes = time[1] - time[0]
            dt_minutes = dt_minutes.seconds / 60
        return 60 / dt_minutes

    @staticmethod
    def can_do(nc):
        can_do = False
        if 'Time' in nc.dimensions:
            can_do = nc.dimensions['Time'].size > 1
        elif 'time' in nc.dimensions:
            can_do = nc.dimensions['time'].size > 1
        return can_do

    def __getitem__(self, item):

        # take care of ellipsis and other strange indexes
        item = list(indexing.expanded_indexer(item, len(self.dimensions)))

        # time is always going to be first dim I hope
        sl = item[0]
        was_scalar = False
        if np.isscalar(sl) and not isinstance(sl, slice):
            was_scalar = True
            sl = slice(sl, sl+1)

        # Ok, get the indexes right
        start = sl.start or 0
        stop = sl.stop or self._nel
        if stop < 0:
            stop += self._nel-1
        start -= 1
        do_nan = False
        if start < 0:
            do_nan = True
        itemr = copy.deepcopy(item)
        item[0] = slice(start, stop-1)
        itemr[0] = slice(start+1, stop)

        # done
        with ScaledVar(self.nc.variables[self.accvn]) as var:
            if do_nan:
                item[0] = slice(0, stop-1)
                out = var[itemr]
                try:
                    # in case we have a masked array
                    out.unshare_mask()
                except:
                    pass
                out[1:, ...] -= var[item]
                out[0, ...] = np.NaN
            else:
                out = var[itemr]
                out -= var[item]
        if was_scalar:
            out = out[0, ...]
        return out * self._factor


class PRCP_NC(AccumulatedVariable):

    def __init__(self, nc):
        AccumulatedVariable.__init__(self, nc, 'RAINNC')
        self.units = 'mm h-1'
        self.description = 'Precipitation rate from grid scale physics'

    @staticmethod
    def can_do(nc):
        return AccumulatedVariable.can_do(nc) and 'RAINNC' in nc.variables


class PRCP_C(AccumulatedVariable):

    def __init__(self, nc):
        AccumulatedVariable.__init__(self, nc, 'RAINC')
        self.units = 'mm h-1'
        self.description = 'Precipitation rate from cumulus physics'

    @staticmethod
    def can_do(nc):
        return AccumulatedVariable.can_do(nc) and 'RAINC' in nc.variables


class PRCP(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['RAINC'])
        self.units = 'mm h-1'
        self.description = 'Total precipitation rate'

    @staticmethod
    def can_do(nc):
        return AccumulatedVariable.can_do(nc) and \
               'RAINC' in nc.variables and 'RAINNC' in nc.variables

    def __getitem__(self, item):
        with ScaledVar(self.nc.variables['PRCP_NC']) as p1, \
                ScaledVar(self.nc.variables['PRCP_C']) as p2:
            return p1[item] + p2[item]


class THETA(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['T'])
        self.units = 'K'
        self.description = 'Potential temperature'

    @staticmethod
    def can_do(nc):
        return 'T' in nc.variables

    def __getitem__(self, item):
        with ScaledVar(self.nc.variables['T']) as var:
            return var[item] + 300.


class TK(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['T'])
        self.units = 'K'
        self.description = 'Temperature'

    @staticmethod
    def can_do(nc):
        return np.all([n in nc.variables for n in ['T', 'P', 'PB']])

    def __getitem__(self, item):
        p1000mb = 100000.
        r_d = 287.04
        cp = 7 * r_d / 2.
        with ScaledVar(self.nc.variables['T']) as var:
            t = var[item] + 300.
        with ScaledVar(self.nc.variables['P']) as p, \
                ScaledVar(self.nc.variables['PB']) as pb:
            p = p[item] + pb[item]
        return (p/p1000mb)**(r_d/cp) * t


class WS(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['U'])
        self.units = 'm s-1'
        self.description = 'Horizontal wind speed'

    @staticmethod
    def can_do(nc):
        return np.all([n in nc.variables for n in ['U', 'V']])

    def __getitem__(self, item):
        with ScaledVar(self.nc.variables['U']) as var:
            ws = var[item]**2
        with ScaledVar(self.nc.variables['V']) as var:
            ws += var[item]**2
        return np.sqrt(ws)


class PRESSURE(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['P'])
        self.units = 'hPa'
        self.description = 'Full model pressure'

    @staticmethod
    def can_do(nc):
        return np.all([n in nc.variables for n in ['P', 'PB']])

    def __getitem__(self, item):

        with ScaledVar(self.nc.variables['P']) as p, \
                ScaledVar(self.nc.variables['PB']) as pb:
            res = p[item] + pb[item]
            if p.units == 'Pa':
                res /= 100
            elif p.units == 'hPa':
                pass
        return res


class GEOPOTENTIAL(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['PH'])
        self.units = 'm2 s-2'
        self.description = 'Full model geopotential'

    @staticmethod
    def can_do(nc):
        return np.all([n in nc.variables for n in ['PH', 'PHB']])

    def __getitem__(self, item):
        with ScaledVar(self.nc.variables['PH']) as p, \
                ScaledVar(self.nc.variables['PHB']) as pb:
            return p[item] + pb[item]


class Z(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['PH'])
        self.units = 'm'
        self.description = 'Full model height'

    @staticmethod
    def can_do(nc):
        return np.all([n in nc.variables for n in ['PH', 'PHB']])

    def __getitem__(self, item):
        with ScaledVar(self.nc.variables['GEOPOTENTIAL']) as var:
            return var[item] / 9.81


class SLP(FakeVariable):
    def __init__(self, nc):
        FakeVariable.__init__(self, nc)
        self._copy_attrs_from(nc.variables['T2'])
        self.units = 'hPa'
        self.description = 'Sea level pressure'
        dims = list(nc.variables['T'].dimensions)
        self.ds = np.nonzero(['bottom_top' in d for d in dims])[0][0]
        self._ds_shape = nc.variables['T'].shape[self.ds]

    @staticmethod
    def can_do(nc):
        # t2 is for attrs (not elegant)
        need = ['T', 'P', 'PB', 'QVAPOR', 'PH', 'PHB', 'T2']
        return np.all([n in nc.variables for n in need])

    def __getitem__(self, item):

        # take care of ellipsis and other strange indexes
        item = list(indexing.expanded_indexer(item, len(self.dimensions)))
        # we need the empty dims for _ncl_slp() to work
        squeezax = []
        for i, c in enumerate(item):
            if np.isscalar(c) and not isinstance(c, slice):
                item[i] = slice(c, c+1)
                squeezax.append(i)
        # add a slice in the 4th dim
        item.insert(self.ds, slice(0, self._ds_shape+1))
        item = tuple(item)

        # get data
        vars = self.nc.variables
        with ScaledVar(vars['TK']) as var:
            tk = var[item]
        with ScaledVar(vars['P']) as p, ScaledVar(vars['PB']) as pb:
            p = p[item] + pb[item]
        with ScaledVar(vars['QVAPOR']) as var:
            q = var[item]
        with ScaledVar(vars['PH']) as ph, ScaledVar(vars['PHB']) as phb:
            z = (ph[item] + phb[item]) / 9.81
        return np.squeeze(_ncl_slp(z, tk, p, q), axis=tuple(squeezax))

# Diagnostic variable classes in a list
var_classes = [cls.__name__ for cls in vars()['FakeVariable'].__subclasses__()]
var_classes.extend([cls.__name__ for cls in
                    vars()['AccumulatedVariable'].__subclasses__()])
var_classes.remove('AccumulatedVariable')


def _interp1d(args):
    f = interp1d(args[0], args[1], fill_value=args[3],
                 bounds_error=False)
    return f(args[2])


def interp3d(data, zcoord, levels, fill_value=np.NaN,
             use_multiprocessing=True):
    """Interpolate on the first dimension of a 3d var

    Useful for WRF pressure or geopotential levels

    Parameters
    ----------
    data: ndarrad
      3d or 4d array of the data to interpolate
    zcoord: ndarray
      same dims as data, the z coordinates of the data points
    levels: 1darray
      the levels at which to interpolate
    fill_value : np.NaN or 'extrapolate', optional
      how to handle levels below the topography. Default is to mark them
      as invalid, but you might want the have them extrapolated.
    use_multiprocessing: bool
      set to false if, for some reason, you don't want to use mp

    Returns
    -------
    a ndarray, with the first dimension now begin of shape nlevels
    """

    ndims = len(data.shape)
    if ndims == 4:
        out = []
        for d, z in zip(data, zcoord):
            out.append(np.expand_dims(interp3d(d, z, levels,
                                               fill_value=fill_value), 0))
        return np.concatenate(out, axis=0)
    if ndims != 3:
        raise ValueError('ndims must be 3')

    if use_multiprocessing:
        inp = []
        for j in range(data.shape[-2]):
            for i in range(data.shape[-1]):
                inp.append((zcoord[:, j, i], data[:, j, i], levels,
                            fill_value))
        _init_pool()
        out = POOL.map(_interp1d, inp, chunksize=1000)
        out = np.asarray(out).T
        out = out.reshape((len(levels), data.shape[-2], data.shape[-1]))
    else:
        # TODO: there got to be a faster way to do this
        # same problem: http://stackoverflow.com/questions/27622808/
        # fast-3d-interpolation-of-atmospheric-data-in-numpy-scipy
        out = np.zeros((len(levels), data.shape[-2], data.shape[-1]))
        for i in range(data.shape[-1]):
            for j in range(data.shape[-2]):
                f = interp1d(zcoord[:, j, i], data[:, j, i],
                             fill_value=fill_value, bounds_error=False)
                out[:, j, i] = f(levels)
    return out


def _ncl_slp(z, t, p, q):
    """Computes the SLP out of the WRF variables.

    This code has been directly translated from the NCL fortran routine found
    in NCL (wrf_user.f). The NCL license is reproduced in the
    salem/licenses directory.

    Parameters
    ----------
    Z: geopotential height
    T: temp
    P: pressure
    Q: specific humidity
    """

    ndims = len(z.shape)
    if ndims == 4:
        out = []
        for _z, _t, _p, _q in zip(z, t, p, q):
            out.append(np.expand_dims(_ncl_slp(_z, _t, _p, _q), 0))
        return np.concatenate(out, axis=0)

    nx = z.shape[-1]
    ny = z.shape[-2]
    nz = z.shape[-3]

    r = 287.04
    g = 9.81
    gamma = 0.0065
    tc = 273.16 + 17.5
    pconst = 10000.

    #  Find least zeta level that is pconst Pa above the surface.  We
    # later use this level to extrapolate a surface pressure and
    # temperature, which is supposed to reduce the effect of the diurnal
    # heating cycle in the pressure field.

    p0 = p[0, ...]

    level = np.zeros((ny, nx), dtype=np.int) - 1
    for k in np.arange(nz):
        pok = np.nonzero((p[k, ...] < (p0 - pconst)) & (level == -1))
        level[pok] = k

    if np.any(level == -1):
        raise RuntimeError('Error_in_finding_100_hPa_up')  # pragma: no cover

    klo = (level-1).clip(0, nz-1)
    khi = (klo+1).clip(0, nz-1)

    if np.any((klo - khi) == 0):
        raise RuntimeError('Trapping levels are weird.')  # pragma: no cover

    x, y = np.meshgrid(np.arange(nx, dtype=np.int),
                       np.arange(ny, dtype=np.int))

    plo = p[klo, y, x]
    phi = p[khi, y, x]

    zlo = z[klo, y, x]
    zhi = z[khi, y, x]

    tlo = t[klo, y, x]
    thi = t[khi, y, x]

    qlo = q[klo, y, x]
    qhi = q[khi, y, x]

    tlo *= (1. + 0.608 * qlo)
    thi *= (1. + 0.608 * qhi)

    p_at_pconst = p0 - pconst
    t_at_pconst = thi - (thi-tlo) * np.log(p_at_pconst/phi) * np.log(plo/phi)
    z_at_pconst = zhi - (zhi-zlo) * np.log(p_at_pconst/phi) * np.log(plo/phi)
    t_surf = t_at_pconst * ((p0/p_at_pconst)**(gamma*r/g))
    t_sea_level = t_at_pconst + gamma * z_at_pconst

    # If we follow a traditional computation, there is a correction to the
    # sea level temperature if both the surface and sea level
    # temperatures are *too* hot.
    l1 = t_sea_level < tc
    l2 = t_surf <= tc
    l3 = ~l1
    t_sea_level = tc - 0.005 * (t_surf-tc)**2
    pok = np.nonzero(l2 & l3)
    t_sea_level[pok] = tc

    # The grand finale
    z_half_lowest = z[0, ...]

    # Convert to hPa in this step
    return 0.01 * (p0 * np.exp((2.*g*z_half_lowest)/(r*(t_sea_level+t_surf))))


def geogrid_simulator(fpath, do_maps=True, map_kwargs=None):
    """Emulates geogrid.exe, which is useful when defining new WRF domains.

    Parameters
    ----------
    fpath: str
       path to a namelist.wps file
    do_maps: bool
       if you want the simulator to return you maps of the grids as well
    map_kwargs: dict
       kwargs to pass to salem.Map()

    Returns
    -------
    (grids, maps) with:
        - grids: a list of Grids corresponding to the domains
          defined in the namelist
        - maps: a list of maps corresponding to the grids (if do_maps==True)
    """

    with open(fpath) as f:
        lines = f.readlines()

    pargs = dict()
    for l in lines:
        s = l.split('=')
        if len(s) < 2:
            continue
        s0 = s[0].strip().upper()
        s1 = list(filter(None, s[1].strip().replace('\n', '').split(',')))

        if s0 == 'PARENT_ID':
            parent_id = [int(s) for s in s1]
        if s0 == 'PARENT_GRID_RATIO':
            parent_ratio = [int(s) for s in s1]
        if s0 == 'I_PARENT_START':
            i_parent_start = [int(s) for s in s1]
        if s0 == 'J_PARENT_START':
            j_parent_start = [int(s) for s in s1]
        if s0 == 'E_WE':
            e_we = [int(s) for s in s1]
        if s0 == 'E_SN':
            e_sn = [int(s) for s in s1]
        if s0 == 'DX':
            dx = float(s1[0])
        if s0 == 'DY':
            dy = float(s1[0])
        if s0 == 'MAP_PROJ':
            map_proj = s1[0].replace("'", '').strip().upper()
        if s0 == 'REF_LAT':
            pargs['lat_0'] = float(s1[0])
        if s0 == 'REF_LON':
            pargs['ref_lon'] = float(s1[0])
        if s0 == 'TRUELAT1':
            pargs['lat_1'] = float(s1[0])
        if s0 == 'TRUELAT2':
            pargs['lat_2'] = float(s1[0])
        if s0 == 'STAND_LON':
            pargs['lon_0'] = float(s1[0])

    # Sometimes files are not complete
    pargs.setdefault('lon_0', pargs['ref_lon'])

    # define projection
    if map_proj == 'LAMBERT':
        pwrf = '+proj=lcc +lat_1={lat_1} +lat_2={lat_2} ' \
               '+lat_0={lat_0} +lon_0={lon_0} ' \
               '+x_0=0 +y_0=0 +a=6370000 +b=6370000'
        pwrf = pwrf.format(**pargs)
    elif map_proj == 'MERCATOR':
        pwrf = '+proj=merc +lat_ts={lat_1} +lon_0={lon_0} ' \
               '+x_0=0 +y_0=0 +a=6370000 +b=6370000'
        pwrf = pwrf.format(**pargs)
    elif map_proj == 'POLAR':
        pwrf = '+proj=stere +lat_ts={lat_1} +lat_0=90.0 +lon_0={lon_0} ' \
               '+x_0=0 +y_0=0 +a=6370000 +b=6370000'
        pwrf = pwrf.format(**pargs)
    else:
        raise NotImplementedError('WRF proj not implemented yet: '
                                  '{}'.format(map_proj))
    pwrf = gis.check_crs(pwrf)

    # get easting and northings from dom center (probably unnecessary here)
    e, n = gis.transform_proj(wgs84, pwrf, pargs['ref_lon'], pargs['lat_0'])

    # LL corner
    nx, ny = e_we[0]-1, e_sn[0]-1
    x0 = -(nx-1) / 2. * dx + e  # -2 because of staggered grid
    y0 = -(ny-1) / 2. * dy + n

    # parent grid
    grid = gis.Grid(nxny=(nx, ny), x0y0=(x0, y0), dxdy=(dx, dy), proj=pwrf)

    # child grids
    out = [grid]
    for ips, jps, pid, ratio, we, sn in zip(i_parent_start, j_parent_start,
                                            parent_id, parent_ratio,
                                            e_we, e_sn):
        if ips == 1:
            continue
        ips -= 1
        jps -= 1
        we -= 1
        sn -= 1
        nx = we / ratio
        ny = sn / ratio
        if nx != (we / ratio):
            raise RuntimeError('e_we and ratios are incompatible: '
                               '(e_we - 1) / ratio must be integer!')
        if ny != (sn / ratio):
            raise RuntimeError('e_sn and ratios are incompatible: '
                               '(e_sn - 1) / ratio must be integer!')

        prevgrid = out[pid - 1]
        xx, yy = prevgrid.corner_grid.x_coord, prevgrid.corner_grid.y_coord
        dx = prevgrid.dx / ratio
        dy = prevgrid.dy / ratio
        grid = gis.Grid(nxny=(we, sn),
                        x0y0=(xx[ips], yy[jps]),
                        dxdy=(dx, dy),
                        pixel_ref='corner',
                        proj=pwrf)
        out.append(grid.center_grid)

    maps = None
    if do_maps:
        from salem import Map
        import shapely.geometry as shpg

        if map_kwargs is None:
            map_kwargs = {}

        maps = []
        for i, g in enumerate(out):
            m = Map(g, **map_kwargs)

            for j in range(i+1, len(out)):
                cg = out[j]
                left, right, bottom, top = cg.extent

                s = np.array([(left, bottom), (right, bottom),
                              (right, top), (left, top)])
                l1 = shpg.LinearRing(s)
                m.set_geometry(l1, crs=cg.proj, linewidth=(len(out)-j),
                               zorder=5)

            maps.append(m)

    return out, maps
