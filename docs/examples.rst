
Examples
========

.. ipython:: python

    import numpy as np
    import salem
    from salem import wgs84

    grid = salem.Grid(nxny=(3, 2), dxdy=(1, 1), ll_corner=(6, 49), proj=wgs84)
    x, y = grid.xy_coordinates
    x
    y