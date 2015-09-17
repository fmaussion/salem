# from __future__ import division
# # Builtins
# from functools import partial
# # External libs
# import pyproj
# import numpy as np
# try:
#     from shapely.ops import transform as shapely_transform
# except ImportError:
#     pass
# try:
#     import geopandas as gpd
# except ImportError:
#     pass
#
# # Locals
# from salem import wgs84, transform_proj, check_crs, Grid
#
# def _secure_shapely(func, geom):
#
#     import sys
#     if sys.version_info[0] < 3:
#         from itertools import izip
#     else:
#         izip = zip
#
#
#     if geom.is_empty:
#         return geom
#     if geom.type in ('Point', 'LineString', 'LinearRing', 'Polygon'):
#
#         # First we try to apply func to x, y, z sequences. When func is
#         # optimized for sequences, this is the fastest, though zipping
#         # the results up to go back into the geometry constructors adds
#         # extra cost.
#         if geom.type in ('Point', 'LineString', 'LinearRing'):
#             return type(geom)(zip(*func(*izip(*geom.coords))))
#         elif geom.type == 'Polygon':
#             shell = type(geom.exterior)(
#                 zip(*func(*izip(*geom.exterior.coords))))
#             holes = list(type(ring)(zip(*func(*izip(*ring.coords))))
#                          for ring in geom.interiors)
#             holes = [h for h in holes if not h.is_empty]
#             return type(geom)(shell, holes)
#
#     elif geom.type.startswith('Multi') or geom.type == 'GeometryCollection':
#         return type(geom)([_secure_shapely(func, part) for part in geom.geoms])
#     else:
#         raise ValueError('Type %r not recognized' % geom.type)
#
#
# def _secure_transform(gdf, to_crs=wgs84):
#
#     from_crs = check_crs(gdf.crs)
#     to_crs = check_crs(to_crs)
#
#     if isinstance(to_crs, pyproj.Proj) and isinstance(from_crs, pyproj.Proj):
#         _project = partial(transform_proj, from_crs, to_crs)
#     elif isinstance(to_crs, Grid):
#         _project = partial(to_crs.transform, crs=from_crs)
#     else:
#         raise NotImplementedError()
#
#     def project(x, y):
#         x, y = _project(x, y)
#         x, y = np.atleast_1d(x), np.atleast_1d(y)
#         ok = np.isfinite(x) & np.isfinite(y)
#         x, y = x[ok], y[ok]
#         if len(x) == 0:
#             x, y = [-999., -999., -999.], [-999., -999., -999.]
#         return x, y
#
#     # Do the job and set the new attributes
#     result = []
#     for geom in gdf.geometry:
#         t = _secure_shapely(project, geom)
#         result.append(t)
#     result = gpd.GeoSeries(result)
#     result.crs = to_crs
#     out = gpd.GeoDataFrame(geometry=result)
#     out.crs = to_crs
#     return out
#
#
# def transform_geopandas(gdf, to_crs=wgs84, inplace=True,
#                         secure_transform=False):
#     """Reprojects a geopandas dataframe.
#
#     Parameters
#     ----------
#     gdf: geopandas dataframe (must have a crs attribute)
#     to_crs: the desired crs
#
#     Returns
#     -------
#     A projected dataframe
#     """
#
#     if secure_transform:
#         return _secure_transform(gdf, to_crs=to_crs)
#
#     from_crs = check_crs(gdf.crs)
#     to_crs = check_crs(to_crs)
#
#     if inplace:
#         out = gdf
#     else:
#         out = gdf.copy()
#
#     if isinstance(to_crs, pyproj.Proj) and isinstance(from_crs, pyproj.Proj):
#         project = partial(transform_proj, from_crs, to_crs)
#     elif isinstance(to_crs, Grid):
#         project = partial(to_crs.transform, crs=from_crs)
#     else:
#         raise NotImplementedError()
#
#     # Do the job and set the new attributes
#     result = out.geometry.apply(lambda geom: shapely_transform(project, geom))
#     result.__class__ = gpd.GeoSeries
#     result.crs = to_crs
#     out.geometry = result
#     out.crs = to_crs
#
#     return out