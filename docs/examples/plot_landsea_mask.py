# -*- coding: utf-8 -*-
"""
===============
Shape to raster
===============

Compute a land/sea mask from a shapefile.

This example illustrates the two different methods available to compute a
raster mask from shapefile polygons.
"""

import salem
import matplotlib.pyplot as plt

# make a local grid from which we will compute the mask
# we make it coarse so that we see the grid points
grid = salem.Grid(proj=salem.wgs84, x0y0=(-18, 3), nxny=(25, 15), dxdy=(1, 1))

# read the ocean shapefile (data from http://www.naturalearthdata.com)
oceans = salem.read_shapefile(salem.get_demo_file('ne_50m_ocean.shp'),
                              cached=True)

# read the lake shapefile (data from http://www.naturalearthdata.com)
lakes = salem.read_shapefile(salem.get_demo_file('ne_50m_lakes.shp'),
                              cached=True)

# The default is to keep only the pixels which center is within the polygon:
mask_default = grid.region_of_interest(shape=oceans)
mask_default = grid.region_of_interest(shape=lakes, roi=mask_default)

# But we can also compute a mask from all touched pixels
mask_all_touched = grid.region_of_interest(shape=oceans, all_touched=True)
mask_all_touched = grid.region_of_interest(shape=lakes, all_touched=True,
                                           roi=mask_all_touched)

# Make a map to check our results
sm = salem.Map(grid, countries=False)
sm.set_shapefile(oceans, edgecolor='k', facecolor='none', linewidth=2)
sm.set_shapefile(lakes, edgecolor='k', facecolor='none', linewidth=2)
sm.set_plot_params(cmap='Blues', vmax=2)

# prepare the figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

# plot 1
sm.set_data(mask_default)
sm.visualize(ax=ax1, addcbar=False, title='Default')
# plot 2
sm.set_data(mask_all_touched)
sm.visualize(ax=ax2, addcbar=False, title='All touched')

# plot!
plt.tight_layout()
plt.show()
