# -*- coding: utf-8 -*-
"""
==============
Add topography
==============

Here we add topography to the plot.

"""

from salem import mercator_grid, Map, get_demo_file
import matplotlib.pyplot as plt

grid = mercator_grid(center_ll=(10.76, 46.798444),
                     extent=(10000, 7000))
c = Map(grid, countries=False)
c.set_lonlat_contours(interval=10)
c.set_shapefile(get_demo_file('Hintereisferner_UTM.shp'))
c.set_topography(get_demo_file('hef_srtm.tif'))

fig, ax = plt.subplots(1, 1)
c.visualize(ax=ax, addcbar=False, title='Default: spline deg 3')
plt.tight_layout()
plt.show()
