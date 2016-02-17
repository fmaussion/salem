"""Diagnostic variables for WRF output.

Diagnostic variables are simply a subclass of FakeVariable that implement
__getitem__. See examples below.
"""
import copy

import numpy as np
try:
    from xarray.core import indexing
except ImportError:
    pass

# TODO: uniformize the interface with "nc" as solo argument everywhere,
# not just ncvars

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
        # Needed later
        self._ds_shape = ncvar.shape[self.ds]

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

        # Horrible logic here. I;ll have to go back to this when I'm smarter
        item = list(indexing.expanded_indexer(item, len(self.dimensions)))
        for i, c in enumerate(item):
            if np.isscalar(c) and not isinstance(c, slice):
                item[i] = slice(c, c)

        start = item[self.ds].start
        stop = item[self.ds].stop
        if start is None:
            start = 0
        if stop is None:
            stop = self._ds_shape
        if stop < 0:
            stop += self._ds_shape-1
        stop = np.clip(stop+1, 0, self._ds_shape)
        tc1 = slice(start+1, stop)
        tc0 = slice(start, stop-1)
        item1 = copy.deepcopy(item)
        item[self.ds] = tc0
        item1[self.ds] = tc1
        return 0.5*(self.ncvar[item] + self.ncvar[item1])


class FakeVariable(object):
    """Duck NetCDF4.Variable class.

    Only a few (the most important) methods are implemented:
        - __getitem__
    """
    def __init__(self, ncvars):
        self.name = self.__class__.__name__
        self.ncvars = ncvars

    @staticmethod
    def can_do():
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()


class T2C(FakeVariable):
    def __init__(self, ncvars):
        FakeVariable.__init__(self, ncvars)
        self.units = 'C'
        self.description = '2m Temperature'
        self.dimensions = self.ncvars['T2'].dimensions

    @staticmethod
    def can_do(ncvars):
        return 'T2' in ncvars

    def __getitem__(self, item):
        return self.ncvars['T2'][item] - 273.15


class TK(FakeVariable):
    def __init__(self, ncvars):
        FakeVariable.__init__(self, ncvars)
        self.units = 'K'
        self.description = 'Temperature'
        self.dimensions = self.ncvars['T'].dimensions

    @staticmethod
    def can_do(ncvars):
        need = ['T', 'P', 'PB']
        return np.all([n in ncvars for n in need])

    def __getitem__(self, item):
        p1000mb = 100000.
        r_d = 287.04
        cp = 7 * r_d / 2.
        t = self.ncvars['T'][item] + 300.
        p = self.ncvars['P'][item] + self.ncvars['PB'][item]
        return (p/p1000mb)**(r_d/cp) * t


class SLP(FakeVariable):
    def __init__(self, ncvars):
        FakeVariable.__init__(self, ncvars)
        self.units = 'hPa'
        self.description = 'Sea level pressure'
        self.dimensions = self.ncvars['T2'].dimensions

    @staticmethod
    def can_do(ncvars):
        need = ['T', 'P', 'PB', 'QVAPOR', 'PH', 'PHB']
        return np.all([n in ncvars for n in need])

    def __getitem__(self, item):
        tk = self.ncvars['TK'][item]
        p = self.ncvars['P'][item] + self.ncvars['PB'][item]
        q = self.ncvars['QVAPOR'][item]
        z = (self.ncvars['PH'][item] + self.ncvars['PHB'][item]) / 9.81
        return _ncl_slp(z, tk, p, q)

# Diagnostic variable classes in a list
var_classes = [cls.__name__ for cls in vars()['FakeVariable'].__subclasses__()]


def _ncl_slp(z, t, p, q):
    """Computes the SLP out of the WRF variables.

    This code has been directly translated from the NCL fortran routine found
    in NCLS's wrf_user.f, therefore I reproduce their licence agreement below.

    Parameters
    ----------
    Z: geopotential height
    T: temp
    P: pressure
    Q: specific humidity

    NCL Licence
    -----------

    Copyright (C) 2015 University Corporation for Atmospheric Research
    The use of this software is governed by a License Agreement.
    See http://www.ncl.ucar.edu/ for more details.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    Neither the names of NCAR's Computational and Information Systems
    Laboratory, the University Corporation for Atmospheric Research, nor
    the names of its contributors may be used to endorse or promote
    products derived from this Software without specific prior written
    permission.

    Redistributions of source code must retain the above copyright
    notice, this list of conditions, and the disclaimer below.

    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions, and the disclaimer below in the
    documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
    SOFTWARE.
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

    level = np.zeros((nx, ny), dtype=np.int) - 1
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

