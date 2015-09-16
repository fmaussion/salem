from __future__ import division

import unittest
from salem import gis


class TestGIS(unittest.TestCase):

    def test_myfunc(self):
        self.assertTrue(gis.myfunc() == 2)