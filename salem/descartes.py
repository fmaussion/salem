"""Paths and patches

This file is part of the package "descartes" by sgilles,
apparently discontinued today.

https://pypi.org/project/descartes

License: BSD License (BSD)
"""

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from numpy import asarray, concatenate, ones


class Polygon(object):
    # Adapt Shapely or GeoJSON/geo_interface polygons to a common interface
    def __init__(self, context):
        if hasattr(context, 'interiors'):
            self.context = context
        else:
            self.context = getattr(context, '__geo_interface__', context)

    @property
    def geom_type(self):
        return (getattr(self.context, 'geom_type', None)
                or self.context['type'])

    @property
    def exterior(self):
        return (getattr(self.context, 'exterior', None)
                or self.context['coordinates'][0])

    @property
    def interiors(self):
        value = getattr(self.context, 'interiors', None)
        if value is None:
            value = self.context['coordinates'][1:]
        return value


def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""
    this = Polygon(polygon)
    assert this.geom_type == 'Polygon'

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    vertices = concatenate(
        [asarray(this.exterior.coords)[:, :2]]
        + [asarray(r.coords)[:, :2] for r in this.interiors])
    codes = concatenate(
        [coding(this.exterior)]
        + [coding(r) for r in this.interiors])
    return Path(vertices, codes)


def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.
    Example (using Shapely Point and a matplotlib axes):
      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)
    """
    return PathPatch(PolygonPath(polygon), **kwargs)
