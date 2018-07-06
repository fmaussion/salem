# -*- coding: utf-8 -*-
"""
====================
Customize Salem maps
====================

How to change the look of a Map?

"""

import salem
import matplotlib.pyplot as plt

# get the map from a WRF file
ds = salem.open_wrf_dataset(salem.get_demo_file('wrfout_d01.nc'))
smap = ds.salem.get_map(countries=False)

# Change the country borders
smap.set_shapefile(countries=True, color='C3', linewidths=2)

# Add oceans and lakes
smap.set_shapefile(oceans=True)
smap.set_shapefile(rivers=True)
smap.set_shapefile(lakes=True, facecolor='blue', edgecolor='blue')

# Change the lon-lat countour setting
smap.set_lonlat_contours(add_ytick_labels=False, interval=5, linewidths=1.5,
                         linestyles='-', colors='C1')

# Add a scalebar (experimental)
smap.set_scale_bar(location=(0.87, 0.04), add_bbox=True)

# done!
smap.visualize()
plt.show()
