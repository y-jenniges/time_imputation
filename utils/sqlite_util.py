"""SQLite utility helpers adapted from a prior publication:
    Yvonne Jenniges. (2025). y-jenniges/ocean_clustering_and_validation:
    Biogeochemical Ocean Regions - Code Base (v1.0.1).
    Zenodo. https://doi.org/10.5281/zenodo.15827777
"""

import numpy as np


class Median(object):
    """ Aggregate function to compute the median in sqlite. """

    def __init__(self):
        self.median = None
        self.vals = []

    def step(self, value):
        self.vals.append(value)

    def finalize(self):
        self.median = np.median(self.vals)
        return self.median


class Std(object):
    """ Aggregate function to compute the standard deviation in sqlite. """

    def __init__(self):
        self.std = None
        self.vals = []

    def step(self, value):
        self.vals.append(value)

    def finalize(self):
        self.std = np.std(self.vals, ddof=1)
        return self.std
