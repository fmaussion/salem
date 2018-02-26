# -*- coding: utf-8 -*-
"""
==============================
Add a Natural Earth background
==============================

An alternative to Google Static Maps
"""

import salem
import matplotlib.pyplot as plt

# get the map from a predefined grid
grid = salem.mercator_grid(transverse=False, center_ll=(16., 0.),
                           extent=(8e6, 9e6))
smap = salem.Map(grid)

# Add the background (other resolutions include: 'mr', 'hr')
smap.set_rgb(natural_earth='lr')

# done!
smap.visualize()
plt.show()
